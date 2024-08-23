
import torch
import numpy as np
from tqdm import tqdm
import os
from termcolor import colored
import argparse


from losses import ChamferDistance, MaxMixturePrior, summed_L2, LossTracker
from visualization import viz_error_curves, viz_iteration, set_init_plot, viz_final_fit
from utils import (check_scan_prequisites_fit_bm, cleanup, 
                   load_config, save_configs,
                   load_landmarks,load_scan,
                   get_already_fitted_scan_names, get_skipped_scan_names, 
                   initialize_fit_bm_loss_weights, load_loss_weights_config, 
                   print_loss_weights, print_losses, print_params, 
                   process_body_model_path, process_default_dtype, 
                   process_landmarks, process_visualize_steps, 
                   create_results_directory, to_txt,
                   setup_socket, send_to_socket,
                   )
from body_models import BodyModel
from body_parameters import BodyParameters
from datasets import FAUST, CAESAR
from dash_app import run_dash_app_as_subprocess



def fit_body_model(input_dict: dict, cfg: dict):
    """
    Fit a body body model (SMPL/SMPLX) to the input scan using
    data, landmark and prior losses

    :param: input_dict (dict): with keys:
            "name": name of the scan
            "vertices": numpy array (N,3)
            "faces": numpy array (N,3) or None if no faces
            "landmarks": dictionary with keys as landmark names and 
                        values as list [x,y,z] or np.ndarray (3,)
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

    # inputs
    input_name = input_dict["name"]
    input_vertices = input_dict["vertices"]
    input_faces = input_dict["faces"]
    input_landmarks = input_dict["landmarks"]
    input_index = input_dict["scan_index"]

    # process inputs
    input_vertices = torch.from_numpy(input_vertices).type(DEFAULT_DTYPE).unsqueeze(0).cuda()
    input_faces = torch.from_numpy(input_faces).type(DEFAULT_DTYPE) if \
                            (not isinstance(input_faces,type(None))) else None

    landmarks_order = sorted(list(input_landmarks.keys()))
    input_landmarks = np.array([input_landmarks[k] for k in landmarks_order])
    input_landmarks = torch.from_numpy(input_landmarks)
    input_landmarks = input_landmarks.type(DEFAULT_DTYPE).cuda()

    # setup body model
    body_model = BodyModel(cfg)
    body_model.cuda()
    body_model_params = BodyParameters(cfg).cuda()
    body_model_landmark_inds = body_model.landmark_indices(landmarks_order)
    print(f"Using {len(input_landmarks)}/{len(body_model.all_landmark_indices)} landmarks.")


    # configure optimization
    ITERATIONS = cfg['iterations']
    LR = cfg['lr']
    START_LR_DECAY = cfg['start_lr_decay_iteration']
    loss_weights = cfg['loss_weights']
    loss_tracker = LossTracker(loss_weights[0].keys())

    body_optimizer = torch.optim.Adam(body_model_params.parameters(), lr=LR)
    chamfer_distance = ChamferDistance()
    prior = MaxMixturePrior(prior_folder=cfg["prior_path"], num_gaussians=8)
    prior = prior.cuda()



    if VISUALIZE:
        fig = set_init_plot(input_vertices[0].detach().cpu(), 
                            body_model.verts_t_pose.detach().cpu(), 
                            title=f"Fitting ({input_name}) - initial setup")
        send_to_socket(fig, socket, SOCKET_TYPE)



    iterator = tqdm(range(ITERATIONS))
    for i in iterator:

        if VERBOSE: print(colored(f"iteration {i}","red"))

        if i in loss_weights.keys():
            if VERBOSE: print(colored(f"\tChanging loss weights","red"))
            data_loss_weight = loss_weights[i]['data']
            landmark_loss_weight = loss_weights[i]['landmark']
            prior_loss_weight = loss_weights[i]['prior']
            beta_loss_weight = loss_weights[i]['beta']

        if VERBOSE: print_loss_weights(data_loss_weight,landmark_loss_weight,
                                        prior_loss_weight,beta_loss_weight,
                                        "loss weights:")

        # forward
        pose, beta, trans, scale = body_model_params.forward()
        if VERBOSE: print_params(pose,beta,trans,scale)
        body_model_verts = body_model.deform_verts(pose,
                                                   beta,
                                                   trans,
                                                   scale)

        # compute losses
        dist1, dist2, _ , _ = chamfer_distance(body_model_verts.unsqueeze(0), input_vertices)
        data_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        data_loss_weighted = data_loss_weight * data_loss
        landmark_loss = summed_L2(body_model_verts[body_model_landmark_inds,:], input_landmarks)
        landmark_loss_weighted = landmark_loss_weight * landmark_loss
        prior_loss = prior.forward(pose[:, 3:], beta)
        prior_loss_weighted = prior_loss_weight * prior_loss
        beta_loss = (beta**2).mean()
        beta_loss_weighted = beta_loss_weight * beta_loss
        loss = data_loss_weighted + landmark_loss_weighted + prior_loss_weighted + beta_loss_weighted

        loss_tracker.update({"data": data_loss_weighted,
                            "landmark": landmark_loss_weighted,
                            "prior": prior_loss_weighted,
                            "beta": beta_loss_weighted,
                            "total": loss})
        
        if VERBOSE: 
            print_losses(data_loss,landmark_loss,prior_loss,beta_loss,"losses")
            print_losses(data_loss_weighted,landmark_loss_weighted,
                         prior_loss_weighted,beta_loss_weighted,"losses weighted")
        iterator.set_description(f"Loss {loss.item():.4f}")

        # optimize
        body_optimizer.zero_grad()
        loss.backward()
        body_optimizer.step()

        if i >= START_LR_DECAY:
            for param_group in body_optimizer.param_groups:
                param_group['lr'] = LR*(ITERATIONS-i)/ITERATIONS
            if VERBOSE: print(colored(f"\tlr: {param_group['lr']}","yellow"))

        if VISUALIZE and (i in VISUALIZE_STEPS):
            new_title = f"Fitting {input_name} - iteration {i}"
            fig = viz_iteration(fig, body_model_verts.detach().cpu(), i , new_title)
            send_to_socket(fig, socket, SOCKET_TYPE)

            new_title = f"Fitting {input_name} losses - iteration {i}"
            fig_losses = viz_error_curves(loss_tracker.losses, loss_weights, 
                                          new_title, VISUALIZE_LOGSCALE)
            send_to_socket(fig_losses, socket, SOCKET_TYPE)


    if VISUALIZE:
        fig = viz_final_fit(input_vertices[0].detach().cpu(), 
                          body_model_verts.detach().cpu(),
                          input_faces,
                          title=f"Fitting {input_name} - final fit")
        send_to_socket(fig, socket, SOCKET_TYPE)

    with torch.no_grad():
        pose, beta, trans, scale = body_model_params.forward()
        body_model_verts = body_model.deform_verts(pose, beta, trans, scale)
        fitted_body_model_verts = body_model_verts.detach().cpu().data.numpy()
        fitted_pose = pose.detach().cpu().numpy()
        fitted_shape = beta.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()

        save_to = os.path.join(SAVE_PATH,f"{input_name}.npz")
        np.savez(save_to, 
                 body_model = body_model.body_model_name,
                 vertices=fitted_body_model_verts, 
                 pose=fitted_pose, 
                 shape=fitted_shape, 
                 trans=trans, 
                 scale=scale, 
                 name=input_name,
                 scan_index=input_index)


def fit_body_model_onto_dataset(cfg: dict):
    
    # get dataset
    dataset_name = cfg["dataset_name"]
    cfg_dataset = cfg[dataset_name]
    cfg_dataset["use_landmarks"] = cfg["use_landmarks"]
    dataset = eval(cfg["dataset_name"])(**cfg_dataset)

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
            continue
        
        process_scan = check_scan_prequisites_fit_bm(input_example)
        if process_scan:
            input_example["scan_index"] = i
            fit_body_model(input_example, cfg)
        else:
            skipped_scans.append(input_example["name"])
            to_txt(skipped_scans, cfg["save_path"], "skipped_scans.txt")
        print(wait_after_fit_func_text)
        wait_after_fit_func("-----------------------------------")
    print(f"Fitting for {dataset_name} dataset completed!")


def fit_body_model_onto_scan(cfg: dict):
    
    wait_after_fit_func = input if cfg["pause_script_after_fitting"] else print

    scan_name = cfg["scan_path"].split("/")[-1].split(".")[0]
    scan_vertices, scan_faces = load_scan(cfg["scan_path"])
    scan_vertices = scan_vertices / cfg["scale_scan"]

    landmarks = load_landmarks(cfg["landmark_path"],
                              cfg["use_landmarks"],
                              scan_vertices)
    landmarks = {lm_name: (np.array(lm_coord) / cfg["scale_landmarks"]).tolist()
                 for lm_name, lm_coord in landmarks.items()}

    input_example = {"name": scan_name,
                    "vertices": scan_vertices,
                    "faces": scan_faces,
                    "landmarks": landmarks,
                    }

    process_scan = check_scan_prequisites_fit_bm(input_example)
    if process_scan:
        input_example["scan_index"] = 0
        fit_body_model(input_example, cfg)
   
        print(f"Fitting completed - press any key to continue!")
        wait_after_fit_func("-----------------------------------")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Subparsers determine the fitting mode: onto_scan or onto_dataset.")
    
    parser_scan = subparsers.add_parser('onto_scan')
    parser_scan.add_argument("--scan_path", type=str, required=True)
    parser_scan.add_argument("--scale_scan", type=float, default=1.0, 
                             help="Scale (divide) the scan vertices by this factor.")
    parser_scan.add_argument("--landmark_path", type=str, required=True)
    parser_scan.add_argument("--scale_landmarks", type=float, default=1.0,
                             help="Scale (divide) the scan landmarks by this factor.")
    parser_scan.set_defaults(func=fit_body_model_onto_scan)

    parser_dataset = subparsers.add_parser('onto_dataset')
    parser_dataset.add_argument("-D","--dataset_name", type=str, required=True)
    parser_dataset.add_argument("-C", "--continue_run", type=str, default=None,
        help="Path to results folder of YYYY_MM_DD_HH_MM_SS format to continue fitting.")
    parser_dataset.set_defaults(func=fit_body_model_onto_dataset)
    
    args = parser.parse_args()


    # load configs
    cfg = load_config()
    cfg_optimization = cfg["fit_body_model_optimization"]
    cfg_datasets = cfg["datasets"]
    cfg_paths = cfg["paths"]
    cfg_general = cfg["general"]
    cfg_web_visualization = cfg["web_visualization"]
    cfg_loss_weights = load_loss_weights_config(
            which_strategy="fit_bm_loss_weight_strategy",
            which_option=cfg_optimization["loss_weight_option"])
    cfg_loss_weights = initialize_fit_bm_loss_weights(cfg_loss_weights)

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
    cfg = process_landmarks(cfg)
    cfg = process_body_model_path(cfg)

    # check if landmarks to use are defined
    assert cfg["use_landmarks"] != [], "Please define landmarks to use in config file!"

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