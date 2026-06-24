"""
Knowledge distillation : AASIST (teacher) → AASIST-L (student).

Loss = alpha * CE(student, hard_label)
     + (1 - alpha) * KL(student_soft_T || teacher_soft_T)

où soft_T = softmax(logits / T) avec température T > 1.
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def get_model(model_config: Dict, device: torch.device):
    module = import_module("models.{}".format(model_config["architecture"]))
    model = getattr(module, "Model")(model_config).to(device)
    nb_params = sum(p.view(-1).size()[0] for p in model.parameters())
    print("  params: {:,}".format(nb_params))
    return model


def get_loader(database_path: Path, seed: int, config: dict):
    track = config["track"]
    prefix = "ASVspoof2019.{}".format(track)

    trn_db = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_db = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_db = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list = database_path / "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(track, prefix)
    dev_trial = database_path / "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(track, prefix)
    eval_trial = database_path / "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix)

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list, is_train=True, is_eval=False)
    d_label_dev, file_dev   = genSpoof_list(dir_meta=dev_trial, is_train=False, is_eval=False)
    file_eval               = genSpoof_list(dir_meta=eval_trial, is_train=False, is_eval=True)

    gen = torch.Generator()
    gen.manual_seed(seed)

    trn_set  = Dataset_ASVspoof2019_train(file_train, d_label_trn, trn_db)
    trn_loader = DataLoader(trn_set, batch_size=config["batch_size"],
                            shuffle=True, drop_last=True, pin_memory=True,
                            worker_init_fn=seed_worker, generator=gen)

    dev_set  = Dataset_ASVspoof2019_devNeval(file_dev, dev_db)
    dev_loader = DataLoader(dev_set, batch_size=config["batch_size"],
                            shuffle=False, drop_last=False, pin_memory=True)

    eval_set = Dataset_ASVspoof2019_devNeval(file_eval, eval_db)
    eval_loader = DataLoader(eval_set, batch_size=config["batch_size"],
                             shuffle=False, drop_last=False, pin_memory=True)

    return trn_loader, dev_loader, eval_loader, dev_trial, eval_trial


def produce_evaluation_file(loader, model, device, score_path, trial_path):
    model.eval()

    # utt_id -> (attaque, bonafide/spoof) à partir du protocole, pour écrire
    # un fichier de scores au format à 4 colonnes attendu par calculate_tDCF_EER
    meta = {}
    with open(trial_path) as f:
        for line in f:
            parts = line.strip().split()
            utt_id, src, key = parts[1], parts[3], parts[4]
            meta[utt_id] = (src, key)

    lines = []
    with torch.no_grad():
        for batch_x, utt_id in loader:
            batch_x = batch_x.to(device)
            _, batch_out = model(batch_x)
            batch_score = batch_out[:, 1].detach().cpu().numpy()
            for u, s in zip(utt_id, batch_score):
                src, key = meta[u]
                lines.append("{} {} {} {}".format(u, src, key, s))
    with open(score_path, "w") as f:
        f.write("\n".join(lines))


def distill_epoch(trn_loader, teacher, student, optimizer, device,
                  scheduler, alpha, temperature, optim_config):
    student.train()
    teacher.eval()

    ce_weight = torch.FloatTensor([0.1, 0.9]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weight)

    running_loss = 0.0
    num_total = 0.0

    for batch_x, batch_y in trn_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size

        # teacher forward (no grad)
        with torch.no_grad():
            _, teacher_logits = teacher(batch_x)

        # student forward
        _, student_logits = student(batch_x)

        # hard loss
        loss_ce = ce_loss(student_logits, batch_y)

        # soft loss (KL avec température)
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        # T² pour compenser la mise à l'échelle du gradient (Hinton et al. 2015)
        loss_kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

        loss = alpha * loss_ce + (1.0 - alpha) * loss_kl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * batch_size

        if optim_config["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()

    return running_loss / num_total


def main(args):
    with open(args.config, "r") as f:
        config = json.loads(f.read())

    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    set_seed(args.seed, config)

    database_path = Path(config["database_path"])
    output_dir    = Path(args.output_dir)

    model_tag = output_dir / "LA_AASIST_distill_ep{}_bs{}".format(
        config["num_epochs"], config["batch_size"])
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")
    writer = SummaryWriter(model_tag)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("GPU not detected!")
    print("Device: {}".format(device))

    # --- Teacher (AASIST, gelé) ---
    print("Chargement du teacher AASIST...")
    teacher = get_model(config["teacher_model_config"], device)
    teacher.load_state_dict(torch.load(config["teacher_path"], map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("Teacher chargé et gelé.")

    # --- Student (AASIST-L, entraîné from scratch) ---
    print("Initialisation du student AASIST-L...")
    student = get_model(config["student_model_config"], device)

    trn_loader, dev_loader, eval_loader, dev_trial, eval_trial = get_loader(
        database_path, args.seed, config)

    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(student.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    alpha       = config.get("distill_alpha", 0.5)
    temperature = config.get("distill_temperature", 4.0)
    print("alpha={}, T={}".format(alpha, temperature))

    best_dev_eer  = 1.0
    best_eval_eer = 100.0
    best_eval_tdcf = 1.0
    best_dev_tdcf  = 0.05
    n_swa_update   = 0

    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    for epoch in range(config["num_epochs"]):
        print("Epoch {:03d}".format(epoch))
        running_loss = distill_epoch(
            trn_loader, teacher, student, optimizer, device,
            scheduler, alpha, temperature, optim_config)

        produce_evaluation_file(dev_loader, student, device,
                                metric_path / "dev_score.txt", dev_trial)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path / "dev_score.txt",
            asv_score_file=database_path / config["asv_score_path"],
            output_file=metric_path / "dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("  loss={:.5f}  dev_eer={:.3f}%  dev_tdcf={:.5f}".format(
            running_loss, dev_eer, dev_tdcf))

        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            best_dev_eer = dev_eer
            print("  → best dev model, epoch {}".format(epoch))
            torch.save(student.state_dict(),
                       model_save_path / "epoch_{}_{:.3f}.pth".format(epoch, dev_eer))

            produce_evaluation_file(eval_loader, student, device,
                                    eval_score_path, eval_trial)
            eval_eer, eval_tdcf = calculate_tDCF_EER(
                cm_scores_file=eval_score_path,
                asv_score_file=database_path / config["asv_score_path"],
                output_file=metric_path / "t-DCF_EER_{:03d}epo.txt".format(epoch),
                printout=False)

            log_text = ""
            if eval_eer < best_eval_eer:
                log_text += "epoch{:03d}, best eer, {:.4f}%".format(epoch, eval_eer)
                best_eval_eer = eval_eer
            if eval_tdcf < best_eval_tdcf:
                log_text += "  best tdcf, {:.4f}".format(eval_tdcf)
                best_eval_tdcf = eval_tdcf
                torch.save(student.state_dict(), model_save_path / "best.pth")
            if log_text:
                print(" ", log_text)
                f_log.write(log_text + "\n")

            optimizer_swa.update_swa()
            n_swa_update += 1

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Évaluation finale (SWA)...")
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, student, device=device)
    produce_evaluation_file(eval_loader, student, device, eval_score_path, eval_trial)
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_tag / "t-DCF_EER.txt")
    f_log.write("SWA — EER: {:.4f}%  tDCF: {:.5f}\n".format(eval_eer, eval_tdcf))
    f_log.close()
    print("Terminé. EER final (SWA): {:.4f}%".format(eval_eer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./exp_result")
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
