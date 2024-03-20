
import argparse
import torch
import numpy as np
import os
from termcolor import colored
from tqdm import tqdm

from body_models import BodyModel
from dash_app import run_dash_app_as_subprocess
from utils import (check_scan_prequisites_fit_verts, cleanup, 
                   create_results_directory, exit_fitting_vertices, 
                   get_already_fitted_scan_names, 
                   get_normals, get_skipped_scan_names, initialize_A, 
                   initialize_fit_verts_loss_weights, load_config, load_landmarks,
                   load_loss_weights_config, load_scan, process_body_model_fit_verts, 
                   process_body_model_path, process_dataset_name, process_default_dtype, 
                   process_landmarks, process_visualize_steps, rotate_points_homo, 
                   save_configs, send_to_socket, setup_socket,
                   set_seed, to_txt, update_normals)
import losses
from visualization import set_init_plot, viz_error_curves, viz_iteration, viz_final_fit
from datasets import CAESAR, FAUST


def fit_vertices(input_dict: dict, cfg: dict):
    """
    Fit template vertices onto scan.
    Either start from body model in T-pose or from previous fit.

    :param: input_dict (dict): with keys:
            "name": name of the scan
            "vertices": numpy array (N,3)
            "faces": numpy array (N,3) or None if no faces
            "landmarks": dictionary with keys as landmark names and 
                        values as list [x,y,z] or np.ndarray (3,)
            "scan_index": (int) index of scan
    :param: cfg (dict): config file defined in configs/config.yaml
    """
    
    DEFAULT_DTYPE = cfg['default_dtype']
    VERBOSE = cfg['verbose']
    VISUALIZE = cfg['visualize']
    VISUALIZE_STEPS = cfg['visualize_steps']
    VISUALIZE_LOGSCALE = cfg["error_curves_logscale"]
    SAVE_PATH = cfg['save_path']
    SOCKET_TYPE = cfg["socket_type"]
    
    if VISUALIZE:
        socket = cfg["socket"]

    # set scan data
    scan_name = input_dict["name"]
    scan_vertices = input_dict["vertices"]
    scan_landmarks = input_dict["landmarks"] if "landmarks" in input_dict.keys() else None
    USE_LANDMARKS = False if isinstance(scan_landmarks,type(None)) else True
    scan_index = input_dict.get("scan_index", 0)
    
    scan_vertices = torch.from_numpy(scan_vertices).type(DEFAULT_DTYPE).unsqueeze(0).cuda()
    if USE_LANDMARKS:
        landmarks_order = sorted(list(scan_landmarks.keys()))
        scan_landmarks = np.array([scan_landmarks[k] for k in landmarks_order])
        scan_landmarks = torch.from_numpy(scan_landmarks)
        scan_landmarks = scan_landmarks.type(DEFAULT_DTYPE).cuda()
    scan_normals = get_normals(scan_vertices[0].detach().cpu())
    scan_normals = scan_normals.cuda()


    # set template data
    # start from body model or from previous fit
    if not isinstance(cfg["start_from_previous_results"], type(None)):
        fit_path = os.path.join(cfg["start_from_previous_results"], 
                                f"{scan_name}.npz")
        if not os.path.exists(fit_path):
            print(colored(f"No previous fit found for scan {scan_name}. Skipping example.","red"))
            return
        template_dict = np.load(fit_path)
        
        template_vertices = template_dict["vertices"]
        template_vertices = torch.from_numpy(template_vertices).type(DEFAULT_DTYPE).unsqueeze(0).cuda()
        
        body_model = BodyModel(cfg)
    else:
        if isinstance(cfg["start_from_body_model"], type(None)):
            cfg["start_from_body_model"] = "smpl"
        body_model = BodyModel(cfg)
        
        template_vertices = body_model.verts_t_pose.unsqueeze(0).cuda() # (1,N,3)

    if USE_LANDMARKS:
        template_landmark_inds = body_model.landmark_indices(landmarks_order)
        print(f"Using {len(scan_landmarks)}/{len(body_model.all_landmark_indices)} landmarks.")
    

    template_normals = get_normals(template_vertices[0].detach().cpu())
    template_normals = template_normals.cuda()
    template_vertices_N = template_vertices.shape[1]

    # visualize starting fitting point
    if VISUALIZE:
        fig = set_init_plot(scan_vertices[0].detach().cpu(), 
                            template_vertices[0].detach().cpu(), 
                            title=f"Fitting ({scan_name}) - initial setup")
        send_to_socket(fig, socket, SOCKET_TYPE)


    # set optimization parameters
    MAX_ITERATIONS = cfg['max_iterations']
    LR = cfg['lr']
    STOP_AT_LOSS_VALUE = float(cfg["stop_at_loss_value"])
    STOP_AT_LOSS_DIFFERENCE = float(cfg["stop_at_loss_difference"])
    A = initialize_A(template_vertices_N, cfg["random_init_A"])
    A = A.cuda()
    optimizer = torch.optim.LBFGS([A], lr=LR)
    loss_func = losses.Losses(cfg, cfg["loss_weights"])
    transform_points = rotate_points_homo

    loss_current = torch.Tensor([10]).cuda()
    loss_previous = torch.Tensor([100]).cuda()
    global closure_call
    closure_call = 0
    closure_calls = []

    iterator = tqdm(range(MAX_ITERATIONS))
    for iteration in iterator:

        if exit_fitting_vertices(loss_current, loss_previous, 
                                STOP_AT_LOSS_VALUE,STOP_AT_LOSS_DIFFERENCE):
            print("Fitting reached loss convergence.")
            break

        def closure():
            optimizer.zero_grad()
            output = transform_points(template_vertices.squeeze(), A)
            output_landmarks = output[template_landmark_inds,:]
            output_normals = update_normals(template_normals, A)

            loss_dict = dict(scan_vertices=scan_vertices,
                             template_vertices = output.unsqueeze(0),
                             A=A,
                             scan_landmarks=scan_landmarks,
                             template_landmarks=output_landmarks,
                             scan_normals=scan_normals,
                             template_normals=output_normals
                             )
            loss = loss_func(**loss_dict)
            
            global closure_call
            closure_call = closure_call + 1
            loss.backward()
            return loss
                
        loss_func.update_loss_weights(iteration)

        loss_previous = loss_current
        optimizer.step(closure)
        output = transform_points(template_vertices.squeeze(), A)
        loss_current = closure()

        if VISUALIZE and (iteration in VISUALIZE_STEPS):
            new_title = f"Fitting {scan_name} - iteration {iteration}"
            fig = viz_iteration(fig, output.detach().cpu(), iteration, new_title)
            send_to_socket(fig, socket, SOCKET_TYPE)

            new_title = f"Fitting {scan_name} losses - iteration {iteration}"
            fig_losses = viz_error_curves(loss_func.loss_tracker.losses, 
                                          loss_func.loss_weights, 
                                          new_title, VISUALIZE_LOGSCALE)
            send_to_socket(fig_losses, socket, SOCKET_TYPE)

        closure_calls.append(closure_call)
        iterator.set_description(f"Loss {loss_current.item():.4f}")


    if VISUALIZE:
        fig = viz_final_fit(scan_vertices.squeeze().detach().cpu(), 
                          output.squeeze().detach().cpu(),
                          None,
                          title=f"Fitting {scan_name} - final fit")
        send_to_socket(fig, socket, SOCKET_TYPE)

    # save fitting
    fitted_vertices = output.detach().cpu().numpy()
    save_to = os.path.join(SAVE_PATH,f"{scan_name}.npz")
    np.savez(save_to, 
            vertices=fitted_vertices, 
            name=scan_name,
            scan_index=scan_index)
    

def fit_vertices_onto_dataset(cfg: dict):
    
    # get dataset
    dataset_name = cfg["dataset_name"]
    cfg_dataset = cfg[dataset_name]
    cfg_dataset["use_landmarks"] = cfg["use_landmarks"]
    dataset = eval(cfg["dataset_name"])(cfg_dataset)

    wait_after_fit_func = input if cfg["pause_script_after_fitting"] else print
    wait_after_fit_func_text = "Fitting completed - press any key to continue!" \
                        if cfg["pause_script_after_fitting"] else "Fitting completed!"

    # if continuing fitting process, get fitted and skipped scans
    fitted_scans = get_already_fitted_scan_names(cfg)
    skipped_scans = get_skipped_scan_names(cfg)

    for i in range(len(dataset)):
        input_example = dataset[i]
        scan_name = input_example["name"]
        print(f"Fitting scan {scan_name} -----------------")

        if (scan_name in fitted_scans) or \
            (scan_name in skipped_scans):
            print("Skip")
            pass
        
        process_scan = check_scan_prequisites_fit_verts(input_example, cfg)
        if process_scan:
            input_example["scan_index"] = i
            fit_vertices(input_example, cfg)
        else:
            skipped_scans.append(input_example["name"])
            to_txt(skipped_scans, cfg["save_path"], "skipped_scans.txt")
        print(wait_after_fit_func_text)
        wait_after_fit_func("-----------------------------------")
    print(f"Fitting for {dataset_name} dataset completed!")


def fit_vertices_onto_scan(cfg: dict):

    wait_after_fit_func = input if cfg["pause_script_after_fitting"] else print
    wait_after_fit_func_text = "Fitting completed - press any key to continue!" \
                        if cfg["pause_script_after_fitting"] else "Fitting completed!"


    scan_verts, scan_faces = load_scan(cfg["scan_path"])
    scan_landmarks = load_landmarks(cfg["landmark_path"])
    
    input_dict = {}
    input_dict["name"] = os.path.basename(cfg["scan_path"]).split(".")[0]
    input_dict["vertices"] = scan_verts
    input_dict["landmarks"] = scan_landmarks
    input_dict["scan_index"] = 0

    fit_vertices(input_dict, cfg)

    print(wait_after_fit_func_text)
    wait_after_fit_func("-----------------------------------")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Subparsers determine the fitting mode: onto_scan or onto_dataset.")
    
    parser_scan = subparsers.add_parser('onto_scan')
    parser_scan.add_argument("--scan_path", type=str, required=True)
    parser_scan.add_argument("--landmark_path", type=str, required=True)
    parser_scan.add_argument("--start_from_previous_results", type=str, default=None,
        help="Path to fitting folder of YYYY_MM_DD_HH_MM_SS format after running fit_body_model.py")
    parser_scan.add_argument("--start_from_body_model", type=str, default=None,
        help="Name of body model to start fitting from.")
    parser_scan.set_defaults(func=fit_vertices_onto_scan)

    parser_dataset = subparsers.add_parser('onto_dataset')
    parser_dataset.add_argument("-D","--dataset_name", type=str, required=None)
    parser_dataset.add_argument("-C", "--continue_run", type=str, default=None,
        help="Path to results folder of YYYY_MM_DD_HH_MM_SS format to continue fitting.")
    parser_dataset.add_argument("--start_from_previous_results", type=str, default=None,
        help="Path to fitting folder of YYYY_MM_DD_HH_MM_SS format after running fit_body_model.py")
    parser_dataset.add_argument("--start_from_body_model", type=str, default=None,
        help="Name of body model to start fitting from.")
    parser_dataset.set_defaults(func=fit_vertices_onto_dataset)
    
    args = parser.parse_args()

    # load configs
    cfg = load_config()
    cfg_optimization = cfg["fit_vertices_optimization"]
    cfg_datasets = cfg["datasets"]
    cfg_paths = cfg["paths"]
    cfg_general = cfg["general"]
    cfg_web_visualization = cfg["web_visualization"]
    cfg_loss_weights = load_loss_weights_config( 
        "fit_verts_loss_weight_strategy",
        cfg_optimization["loss_weight_option"])
    cfg_loss_weights = initialize_fit_verts_loss_weights(cfg_loss_weights)

    # merge configs
    cfg = {}
    cfg.update(cfg_optimization)
    cfg.update(cfg_datasets)
    cfg.update(cfg_paths)
    cfg.update(cfg_general)
    cfg.update(cfg_web_visualization)
    cfg.update(vars(args))
    cfg["loss_weights"] = cfg_loss_weights
    cfg["continue_run"] = cfg["continue_run"] if "continue_run" in cfg.keys() else None

    # process configs
    cfg["save_path"] = create_results_directory(cfg["save_path"], 
                                                cfg["continue_run"])
    cfg = process_default_dtype(cfg)
    cfg = process_visualize_steps(cfg)
    cfg = process_body_model_fit_verts(cfg)
    cfg = process_body_model_path(cfg)
    cfg = process_landmarks(cfg)
    cfg = process_dataset_name(cfg)
    set_seed(cfg["seed"])

    # save configs into results dir
    save_configs(cfg)

    # create web visualization
    if cfg["visualize"]:
        cfg["socket"] = setup_socket(cfg["socket_type"])
        dash_app_process, dash_app_pid = run_dash_app_as_subprocess(cfg['socket_port'])
        print(f"Fitting visualization on http://localhost:{cfg['socket_port']}/")

    # wrapped in a try-except to make sure that the 
    # web visualization socket is closed properly
    try:
        args.func(cfg)
    except (Exception,KeyboardInterrupt) as e:
        print(e)

    if cfg["visualize"]:
        cleanup(cfg["visualize"], cfg["socket"], dash_app_process, dash_app_pid)