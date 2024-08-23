
import argparse
import os
import numpy as np
import sys
from tqdm import tqdm
from glob import glob 
import torch
from body_models import BodyModel, infer_body_model

from utils import load_config, load_scan, process_body_model_path
from datasets import FAUST, CAESAR
from visualization import visualize_pve

import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path,"pyTorchChamferDistance"))
from chamfer_distance import ChamferDistance


def evalute_pve(fitting_results_path, ground_truth_path, **kwargs):
    """
    Evaluate the PVE of the fitted body model to the ground 
    truth (GT) body. When evaluating a dataset - the GT is presumed to 
    be given by the dataset as noted in the documentation. 
    When evaluating scan - the GT is presumed to be given by the 
    ground_truth_path.

    :param fitting_results_path: (str) Path to folder with fitting results.
    :param ground_truth_path: (str) Path to ground truth body model.
    """
    
    # check if evaluating single scan or dataset
    cfg = load_config(f"{fitting_results_path}/config.yaml")
    VISUALIZE = kwargs["visualize"]
    if VISUALIZE:
        got_body_model_type = False

    # evaluating dataset or scan
    if "onto_dataset" in cfg["func"]:

        mse_errors = []

        # load dataset
        dataset_name = cfg["dataset_name"]
        cfg_dataset = cfg[dataset_name]
        cfg_dataset["use_landmarks"] = cfg["use_landmarks"]
        cfg_dataset["load_gt"] = True
        dataset = eval(cfg["dataset_name"])(**cfg_dataset)

        # chek if gt avaialable
        msg = "Ground truth for this dataset is not available"
        assert dataset.gt_available, msg

        selected_examples = kwargs["select_examples"]

        # iterate over dataset
        missing_fits = 0
        for i in tqdm(range(len(dataset))):
            input_example = dataset[i]
            gt_verts = input_example["vertices_gt"]
            scan_name = input_example["name"]

            if selected_examples:
                if scan_name not in selected_examples:
                    continue

            # check if fit available
            fit_path = os.path.join(fitting_results_path, f"{scan_name}.npz")
            if not os.path.exists(fit_path):
                missing_fits += 1
                continue

            # load fit
            fit = np.load(fit_path)
            fit_verts = fit["vertices"]

            # compute MSE
            dists = np.sqrt(np.sum((fit_verts - gt_verts)**2, axis=1))
            mse = np.mean(dists)

            # save mse
            mse_errors.append(mse)

            if VISUALIZE:
                if not got_body_model_type:
                    cfg = load_config()
                    cfg = cfg["paths"]
                    if "body_model" not in fit.keys():
                        cfg["body_model"] = infer_body_model(fit_verts.shape[0])
                    else:
                        cfg["body_model"] = fit["body_model"].item()
                    cfg = process_body_model_path(cfg)
                    body_model = BodyModel(cfg)
                    body_model_faces = body_model.faces
                    got_body_model_type = True

                visualize_pve(fit_verts, dists, body_model_faces,scan_name)
                input("Press any key to continue...")

        mean_mse = np.mean(mse_errors)
        if missing_fits > 0:
            fitted_examples = len(dataset)-missing_fits
            print(f"Evaluation results for for {fitted_examples}/{len(dataset)} examples")
        print(f"MSE from GT to estimated body model " +
              f"for {dataset_name} dataset: {mean_mse:.4f}")

    # evaluate fitting onto scan
    elif "onto_scan" in cfg["func"]:
        
        gt_verts, _ = load_scan(ground_truth_path)

        fit_path = glob(fitting_results_path + "/*.npz")[0]
        scan_name = fit_path.split("/")[-1].split(".")[0]
        if not os.path.exists(fit_path):
            msg = f"Fit for scan {scan_name} not \
                    available in folder {fitting_results_path}"
            sys.exit(msg)
        fit = np.load(fit_path)
        fit_verts = fit["vertices"]

        dists = np.sqrt(np.sum((fit_verts - gt_verts)**2, axis=1))
        mean_mse = np.mean(dists)

        if VISUALIZE:
            if not got_body_model_type:
                cfg = load_config()
                cfg = cfg["paths"]
                if "body_model" not in fit.keys():
                    cfg["body_model"] = infer_body_model(fit_verts.shape[0])
                else:
                    cfg["body_model"] = fit["body_model"].item()
                cfg = process_body_model_path(cfg)
                body_model = BodyModel(cfg)
                body_model_faces = body_model.faces
                got_body_model_type = True

            visualize_pve(fit_verts, dists, body_model_faces, scan_name)

        print(f"MSE from GT to estimated body model " +
              f"for scan {scan_name}: {mean_mse:.4f}")


def evaluate_chamfer(fitting_results_path, scan_path, device, **kwargs):
    """
    Evaluate the chamfer distance of the fitted body model to the scan.

    
    :param fitting_results_path: (str) Path to folder with fitting results 
                                obtained by the scripts fit_body_model.py or 
                                fit_vertices.py
    :param scan_path: (str) Path to original scan that was fitted. This is 
                        only needed if evaluating a single scan.
    :param device: (str) pytorch device to use for chamfer evaluation. 
                        cpu or cuda
    """
    # check if evaluating single scan or dataset
    cfg = load_config(f"{fitting_results_path}/config.yaml")
    chamfer_distance = ChamferDistance()
    if ("cuda" in device):
        nr_gpus = torch.cuda.device_count()
        selected_device = int(device.split(":")[1])
        if selected_device <= nr_gpus:
            device = torch.device(device)
        else:
            raise ValueError(f"There are {nr_gpus} gpus." + 
                             f"Cant select {selected_device}.")
    else:
        device = torch.device("cpu")
    # device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # cases for evaluating dataset or scan
    if "onto_dataset" in cfg["func"]:
        chamfer_standard = 0
        chamfer_bidirectional_average = 0
        chamfer_bm2scan = 0
        chamfer_scan2bm = 0

        # load dataset
        dataset_name = cfg["dataset_name"]
        cfg_dataset = cfg[dataset_name]
        cfg_dataset["use_landmarks"] = cfg["use_landmarks"]
        cfg_dataset["load_gt"] = False
        dataset = eval(cfg["dataset_name"])(**cfg_dataset)

        N = len(dataset)
        selected_examples = kwargs["select_examples"]
        if not isinstance(selected_examples,type(None)):
            if len(selected_examples) == 1:
                if selected_examples[0].endswith(".txt"):
                    with open(selected_examples[0],"r") as f:
                        selected_examples = f.read()
                    selected_examples = selected_examples.split("\n")
                
        # iterate over dataset
        actual_N = 0
        for i in tqdm(range(N)):
            input_scan = dataset[i]
            scan_vertices = input_scan["vertices"]
            if isinstance(scan_vertices, type(None)):
                continue
            scan_vertices = torch.from_numpy(scan_vertices).unsqueeze(0).float()
            scan_name = input_scan["name"]

            if not isinstance(selected_examples,type(None)):
                if scan_name not in selected_examples:
                    continue

            # check if fit available
            fit_path = os.path.join(fitting_results_path, 
                                    f"{scan_name}.npz")
            if not os.path.exists(fit_path):
                continue

            # load fit
            fit = np.load(fit_path)
            fit_verts = fit["vertices"]
            fit_verts = torch.from_numpy(fit_verts).unsqueeze(0).float()

            # compute chamfer
            # dist 1 is 1 x 6890 - chamfer_distance does not return euclidean distance but squared distance
            dist1, dist2 = chamfer_distance(fit_verts.to(device), scan_vertices.to(device))
            chamfer_standard += (torch.mean(dist1) + torch.mean(dist2)).detach().cpu().item()
            chamfer_bidirectional_average += torch.mean(torch.cat([dist1[0],dist2[0]])).detach().cpu().item()
            chamfer_bm2scan += torch.mean(torch.sqrt(dist1)).detach().cpu().item()
            chamfer_scan2bm += torch.mean(torch.sqrt(dist2)).detach().cpu().item()

            actual_N += 1


        chamfer_standard /= actual_N
        chamfer_bidirectional_average /= actual_N
        chamfer_bm2scan /= actual_N
        chamfer_scan2bm /= actual_N

        print(f"Chamfer distances between scan and fitted body model for {dataset_name} dataset:")
        print(f"N examples: {actual_N}")
        print(f"Chamfer standard: {chamfer_standard:.4f}")
        print(f"Chamfer bidirectional average: {chamfer_bidirectional_average:.4f}")
        print(f"Chamfer from body model to scan: {chamfer_bm2scan:.4f}")    
        print(f"Chamfer from scan to body model: {chamfer_scan2bm:.4f}")       

    elif "onto_scan" in cfg["func"]:
        
        scan_verts, _ = load_scan(scan_path)
        scan_verts = torch.from_numpy(scan_verts).unsqueeze(0).float()

        fit_path = glob(fitting_results_path + "/*.npz")[0]
        scan_name = fit_path.split("/")[-1].split(".")[0]
        if not os.path.exists(fit_path):
            msg = f"Fit for scan {scan_name} not \
                    available in folder {fitting_results_path}"
            sys.exit(msg)
        fit = np.load(fit_path)
        fit_verts = fit["vertices"]
        fit_verts = torch.from_numpy(fit_verts).unsqueeze(0).float()

        # compute chamfer
        dist1, dist2 = chamfer_distance(fit_verts.to(device), scan_verts.to(device))
        chamfer_standard = (torch.mean(dist1) + torch.mean(dist2)).detach().cpu().item()
        chamfer_bidirectional_average = torch.mean(torch.cat([dist1,dist2])).detach().cpu().item()
        chamfer_bm2scan = torch.mean(torch.sqrt(dist1)).detach().cpu().item()
        chamfer_scan2bm = torch.mean(torch.sqrt(dist2)).detach().cpu().item()

        print(f"Chamfer distance between scan and fitted body model for scan {scan_name}")
        print(f"Chamfer standard: {chamfer_standard:.4f}")
        print(f"Chamfer bidirectional average: {chamfer_bidirectional_average:.4f}")
        print(f"Chamfer from body model to scan: {chamfer_bm2scan:.4f}")
        print(f"Chamfer from scan to body model: {chamfer_scan2bm:.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subparsers")

    # evaluate PVE
    parser_pve = subparsers.add_parser('pve')
    parser_pve.add_argument("-F","--fitting_results_path", type=str, required=True)
    parser_pve.add_argument("-G","--ground_truth_path", type=str, default=None,
                               help="Path to ground truth body model. \
                                Only necessary if evaluating fit to a \
                                    single scan (not dataset).")
    parser_pve.add_argument('--select_examples', nargs='+', 
                            help='Select subset of examples from dataset. \
                                Only used when evaluating fit to dataset.', 
                            default=None)
    parser_pve.add_argument("-V","--visualize", action="store_true")
    parser_pve.set_defaults(func=evalute_pve)


    # evaluate chamfer distance to scan
    parser_chamfer = subparsers.add_parser('chamfer')
    parser_chamfer.add_argument("-F","--fitting_results_path", type=str, required=True)
    parser_chamfer.add_argument("-S","--scan_path", type=str, default=None,
                                     help="Path to scan to fit." +
                                           "Only necessary if evaluating fit to a" +
                                            "single scan (not dataset).")
    parser_chamfer.add_argument('--select_examples', nargs='+', 
                                    help='Select subset of examples from dataset. \
                                          Either by name of scans, or with .txt file \
                                          where one scan name by row \
                                        Only used when evaluating fit to dataset.', 
                                    default=None)
    parser_chamfer.add_argument("--device", type=str, default="cpu",
                                     help="Device to use for chamfer evaluation.")
    parser_chamfer.set_defaults(func=evaluate_chamfer)
    
    args = parser.parse_args()

    args.func(**vars(args))