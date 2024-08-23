

from termcolor import colored
import yaml
from datetime import datetime
import os
import torch
from glob import glob
import zmq
import plotly
import json
from dash_app import terminate_dash_app_subprocess
import landmarks as landmarks_module
import json
import gzip
from typing import List
import numpy as np
import open3d as o3d
import random
import tempfile
import trimesh

#############################################################
#                   Configs                                 #
#############################################################

def load_config(path="configs/config.yaml"):
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

def load_paths():
    with open('configs/config.yaml','r') as f:
        paths = yaml.safe_load(f)
    paths = paths['PATHS']

    return paths

def load_loss_weights_config(which_strategy,which_option,path=None):
    if not isinstance(path,type(None)):
        with open(path,"r") as f:
            cfg_weights = yaml.safe_load(f)
    else:
        with open("configs/loss_weight_configs.yaml","r") as f:
            cfg_weights = yaml.safe_load(f)

    cfg_weights = cfg_weights[which_strategy]
    cfg_weights = cfg_weights[which_option]

    return cfg_weights

def initialize_fit_bm_loss_weights(loss_weights: dict):
    """
    Initialize the loss_weights if 0-th iteration is not provided in
    loss_weights dictionary
    :param loss_weights: dictionary of loss weights
                         with keys as iteration numbers
                        and values as dictionaries of loss weights
                        for DATA, LANDMARK, PRIOR, and BETA Losses
    """
    if 0 not in loss_weights.keys():
        # default initial weights
        update_with = {0:{"DATA_LOSS": 0.02,
                        "LANDMARK_LOSS": 5,
                        "PRIOR_LOSS": 0.001,
                        "BETA_LOSS": 0.01}}
        print(f"Setting default initial loss weights to {update_with[0]}")
        print(f"To set your own, check iteration 0 in \
              configs/loss_weight_configs.yaml")
        loss_weights.update(update_with)

    return loss_weights

def initialize_fit_verts_loss_weights(loss_weights: dict):
    """
    Initialize the loss_weights if 0-th iteration is missing from the
    loss_weights dictionary
    :param loss_weights: dictionary of loss weights
                         with keys as iteration numbers
                        and values as dictionaries of loss weights
                        for DATA, LANDMARK, PRIOR, and BETA Losses
    """
    if 0 not in loss_weights.keys():
        lowest_iteration = min(loss_weights.keys())
        update_with = {0:loss_weights[lowest_iteration]}
        loss_weights.update(update_with)

    return loss_weights

def save_configs(cfg: dict):
    save_path = cfg["save_path"]

    config_name = os.path.join(save_path,"config.yaml")
    with open(config_name, 'w') as file:
        _ = json.dump(cfg, file, default=lambda o: str(o))
        
#############################################################
#                   Process configs                         #
#############################################################


def process_visualize_steps(cfg: dict):
    """
    The optimization iteration steps to visualize from visualize_steps 
    are processed into a list of iteration indices to visualize.

    :param cfg: config dictionary with "visualize_steps" key which is a 
                string of sum of ranges and lists of iteration indices 
                to visualize
                example: range(0,300,20) + range(300,500,30) + [499] 
                visualizes
                iterations 0,20,40,...,300,330,360,...,500, and 499
    """

    visualize_steps_string = cfg["visualize_steps"]
    ranges = visualize_steps_string.split("+")
    ranges = [list(eval(x)) for x in ranges] # list of lists
    ranges_flat = sum(ranges,[]) # flatten list of lists
    steps_to_visualize = sorted(ranges_flat)
    cfg["visualize_steps"] = steps_to_visualize

    return cfg

def process_default_dtype(cfg: dict):
    cfg["default_dtype"] = eval(cfg["default_dtype"])
    return cfg

def process_body_model_path(cfg: dict):

    body_model_type = cfg["body_model"].lower()
    cfg["body_models_path"] = os.path.join(cfg["body_models_path"],
                                          body_model_type)

    return cfg

def process_body_model_fit_verts(cfg):

    if not isinstance(cfg["start_from_body_model"],type(None)):
        cfg["body_model"] = cfg["start_from_body_model"]
    
    elif not isinstance(cfg["start_from_previous_results"],type(None)):

        npz_files = glob(os.path.join(cfg["start_from_previous_results"],"*.npz"))
        data = np.load(npz_files[0])
        cfg["body_model"] = data["body_model"].item()
    
    else: 
        cfg["body_model"] = "smpl"

    return cfg

def process_landmarks(cfg: dict):
    """

    This function processes the cfg["use_landmarks"] which states
    which landamrks to use during optimization. The function returns
    a standardized list of landmark names to use defined on the body 
    model. To find all the defined landmarks on the body model, 
    check landmarks.py.
    

    The "use_landmarks" can be defined as
    - a string "all" indicating to use all the possible landmarks, 
    - "none" / None / [] indicating to use no landmarks, 
    - string of the dictionary name from landmarks.py which has 
        landmark:indices mappings for a body model,
    - a list of landmark names to use ["Lt. 10th Rib", "Lt. Dactylion",..]

    :param cfg: config dictionary with "use_landmarks" key
                where "use_landmarks" is defined in description above
    """

    use_lm = cfg["use_landmarks"]

    body_model = cfg["body_model"]
    body_model_defined_landmarks =  f"{body_model.upper()}_INDEX_LANDMARKS"
    possible_landmarks = getattr(landmarks_module,body_model_defined_landmarks)
    possible_lm = list(possible_landmarks.keys())

    # use no landmarks
    if (isinstance(use_lm,type(None)) or 
            (isinstance(use_lm,str) and use_lm.lower() == "none") or 
            use_lm == []):
        use_lm = []

    # use all landamrks
    if isinstance(use_lm,str):
        if use_lm.lower() == "all":
            use_lm = body_model_defined_landmarks
        use_lm = getattr(landmarks_module, use_lm.upper(), None)
        if isinstance(use_lm,type(None)):
            raise ValueError(f"Mapping {use_lm} does not exist. \
                               Check landmarks.py.")
        
        use_lm = list(use_lm.keys())
            

    # use a specific list of landamrks
    # NOTE: includes the case when use_lm was a string of a landmarks dict name
    # parsed in the previous if statement and converted to a list
    if isinstance(use_lm,list) and use_lm != []:

        can_use_lm = len(set(possible_lm)) - len(set(possible_lm) - set(use_lm))

        if can_use_lm == 0:
            msg = "None of the landmarks you provided are defined on the body model. \
                    Please cehck the landmarks.py for the full list of landmarks "
            raise ValueError(msg)


    cfg["use_landmarks"] = use_lm

    return cfg

def process_dataset_name(cfg:dict):

    fitting_func_name = cfg["func"].__name__
    if "onto_scan" in fitting_func_name:
        pass
    elif "onto_dataset" in fitting_func_name:
        condition = "start_from_previous_results" in cfg.keys() \
                    or "dataset_name" in cfg.keys()
        assert condition, "Either start_from_previous_results or dataset_name must be defined."

        # infer from previous results
        if isinstance(cfg["dataset_name"], type(None)):
            cfg_path = os.path.join(cfg["start_from_previous_results"],"config.yaml")
            cfg_tnp = load_config(cfg_path)
            cfg["dataset_name"] = cfg_tnp["dataset_name"]

    return cfg

#############################################################
#                   Results                                 #
#############################################################

def create_results_directory(save_path: str = "/SMPL-Fitting/results",
                             continue_run: str = None,
                             sequences: List[str] = None):
    """
    Save results in save_path/YYYY_MM_DD_HH_MM_SS folder.
    If continue_run is folder of type YYYY_MM_DD_HH_MM_SS, then
    save results in save_path/continue_run folder.
    
    :param save_path: path to save results to
    :param continue_run: string of type YYYY_MM_DD_HH_MM_SS
    """


    if continue_run:
        # check if formatting of continue_run folder looks like "%Y_%m_%d_%H_%M_%S"
        # wil raise ValueError if not
        try:
            _ = datetime.strptime(continue_run.split("/")[-1],"%Y_%m_%d_%H_%M_%S")
        except Exception as e:
            raise ValueError("CONTINUE_RUN must be a folder of type YYYY_MM_DD_HH_MM_SS")

        print(f"Continuing run from previous checkpoint")
        save_path = os.path.join(save_path,continue_run)
    else:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = os.path.join(save_path,current_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    print(f"Saving results to {save_path}")

    if not isinstance(sequences,type(None)):
        for seq in sequences:
            seq_path = os.path.join(save_path,seq)
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)

    return save_path


#############################################################
#                   print/txt                               #
#############################################################

def to_txt(list,path,name):
    with open(os.path.join(path,name),"w+") as f:
        f.write("\n".join(list))

def print_params(pose: torch.tensor,
                 beta:torch.tensor,
                 trans:torch.tensor,
                 scale:torch.tensor):
    print(colored(f"\tpose:","yellow"), f"{pose.tolist()}")
    print(colored(f"\tbeta:","yellow"), f"{beta.tolist()}")
    print(colored(f"\ttrans:","yellow"), f"{trans.tolist()}")
    print(colored(f"\tscale:","yellow"), f"{scale.tolist()}")

def print_losses(data_loss:torch.tensor,
                 landmark_loss:torch.tensor,
                 prior_loss:torch.tensor,
                 beta_loss:torch.tensor,
                 loss_names:str = "losses"):
    
    print_str = colored(f"\t{loss_names}:", "yellow")
    print_str += colored(f" DATA:","blue")
    print_str += colored(f"{data_loss.item():.6f}","green")
    print_str += colored(f", LANDMARK:","blue")
    print_str += colored(f"{landmark_loss.item():.6f}","green")
    print_str += colored(f", PRIOR:","blue")
    print_str += colored(f"{prior_loss.item():.6f}","green")
    print_str += colored(f", BETA:","blue")
    print_str += colored(f"{beta_loss.item():.6f}","green")
    print(print_str)

def print_loss_weights(data_loss:torch.tensor,
                        landmark_loss:torch.tensor,
                        prior_loss:torch.tensor,
                        beta_loss:torch.tensor,
                        loss_names:str = "loss weights:"):
    print_str = colored(f"\t{loss_names}:", "yellow")
    print_str += colored(f" DATA:","blue")
    print_str += colored(f"{data_loss:.4f}","green")
    print_str += colored(f", LANDMARK:","blue")
    print_str += colored(f"{landmark_loss:.4f}","green")
    print_str += colored(f", PRIOR:","blue")
    print_str += colored(f"{prior_loss:.4f}","green")
    print_str += colored(f", BETA:","blue")
    print_str += colored(f"{beta_loss:.4f}","green")
    print(print_str)


#############################################################
#                   Fitting                                 #
#############################################################

def get_already_fitted_scan_names(cfg: dict):
    """
    Return list of already fitted scans - founds as .npz files 
    in the save_path directory

    :param cfg: config dictionary with
                save_path: path where results are saved
    
    :return list of scan names that have already been fitted
    """

    fitted_scans_path = os.path.join(cfg["save_path"],"*.npz")
    fitted_scans = glob(fitted_scans_path)
    fitted_scans = [name.split("/")[-1].split(".")[0] 
                    for name in fitted_scans]

    return fitted_scans

def check_scan_prequisites_fit_bm(input_dict:dict, verbose=True):
    """
    Check if the input_dict has all the required fields with defined
    values. Required data for fitting is the scans: 
    - name, 
    - vertices, 
    - landmarks

    If all the data is there, return True, else False

    :param input_dict: dictionary with keys name, vertices, landmarks
    :param verbose: print message if example will not be processed

    :return (boolean) indicating if input_dict has all data and 
            the fitting can proceed
    """

    input_keys = input_dict.keys()
    expected_keys = ["name","vertices","landmarks"]

    msg = f"Skipping example {input_dict['name']} because of missing data"

    # check if input_dict has all the required keys
    if not set(expected_keys).issubset(set(input_keys)):
        if verbose:
            print(colored(msg,"red"))
        return False

    
    # check if any value from input_dict is None
    # that means that some of the data is missing
    if any(input_dict[key] is None for key in expected_keys):
        if verbose:
            print(colored(msg,"red"))
        return False
    
    return True

def check_scan_prequisites_fit_verts(input_dict:dict, cfg:dict, verbose=True):
    """
    Check if all the scan dict has all the required fields
    If all the data is there, return True, else False
    Required data is: name, vertices

    :param input_example: dictionary with keys 
    :param verbose: print messages if example will not be processed

    :return boolean indicating if example has all data and 
            can be processed
    """

    input_keys = input_dict.keys()
    expected_keys = ["name","vertices"]
    if "landmark" in cfg["use_losses"]:
        expected_keys.append("landmarks")

    msg = f"Skipping example {input_dict['name']} because of missing data"

    # check if input_dict has all the required keys
    if not set(expected_keys).issubset(set(input_keys)):
        if verbose:
            print(colored(msg,"red"))
        return False

    
    # check if input_dict has all the data required to fit the body model
    if any(input_dict[key] is None for key in expected_keys):
        if verbose:
            print(colored(msg,"red"))
        return False
    
    return True

def get_skipped_scan_names(cfg: dict):
    """
    Get list of scan names that have been 
    skipped because of missing data.

    :param cfg: config dictionary with
                save_path: path to save results to

    :return list of scan names that have been skipped
    """
    
    skipped_scans_path = os.path.join(cfg["save_path"],
                                      "skipped_scans.txt")
    if os.path.exists(skipped_scans_path):
        with open(skipped_scans_path, "r") as f:
            skipped_scans = f.read()
        skipped_scans = skipped_scans.split("\n")
    else:
        skipped_scans = []
    
    return skipped_scans


#############################################################
#                   Socket                                  #
#############################################################

def setup_socket(socket_type="zmq"):
    """
    Set up the socket for sending data to the Dash app
    Currently only zmq is supported.

    :param socket_type: type of socket to use
                        options: zmq
    :param port: port to connect to

    :return socket: socket object
    """

    socket_options = ["zmq"]
    socket_options_str = ' or '.join(socket_options)

    if socket_type == "zmq":
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        # Connect to the address where the Dash app is listening
        socket.connect(f"tcp://127.0.0.1:5555")
        return socket
    # elif socket_type == "flask":
    #     # socket = SocketIO(message_queue='redis://')  # Use Redis message queue 
    #     socket = SocketIO()
    #     return socket
    else:
        raise ValueError(f"Socket must be of {socket_options_str} type, got {socket_type}")

def send_to_socket(fig, socket, socket_type="zmq"):
    """
    Send data to socket
    :param fig: plotly figure
    :param socket: socket object
    :param socket_type: type of socket to use
                        options: zmq
    """

    json_figure = plotly.io.to_json(fig)

    if socket_type == "zmq":
        json_data = json.dumps(json_figure)
        socket.send_string(json_data)
    # elif zmq_or_flask == "flask":
    #     # NOTE: this does not work - it says NoneType has no attribute 'emit'
    #     # even though socket and json_figure are not None
    #     # socket.emit('update-plot', json_figure)
    #     pass

def close_sockets(socket):
    """
    Close a socket.
    :param socket: socket object
    """

    # if socket made with zmq
    try:
        socket.close()
    except Exception as e:
        pass
    # if socket made with Flask
    try:
        socket.stop()
    except Exception as e:
        pass


#############################################################
#                   Cleanup                                 #
#############################################################

def cleanup(visualize, socket,dash_app_process,dash_app_pid):
    """
    Close sockets and terminate dash app subprocess.
    :param visualize: boolean indicating if visualization is on
    :param socket: socket object
    :param dash_app_process: dash app subprocess object
    :param dash_app_pid: dash app subprocess pid
    """

    if visualize:
        close_sockets(socket)
        terminate_dash_app_subprocess(dash_app_process,dash_app_pid)


#############################################################
#                   loading                                 #
#############################################################

def parse_landmark_txt_coords_formatting(data: List[str]):
    """
    Parse landamrk txt file with formatting
    x y z landmark_name

    :param data (List[str]) list of strings, each string
                represents one line from the txt file
    
    :return landmarks (dict) with formatting 
                      {landmark_name: [x,y,z]}
    """

    # get number of landmarks
    N = len(data)
    if data[-1] == "\n":
        N -= 1

    # define landmarks
    landmarks = {}

    for i in range(N):
        splitted_line = data[i].split(" ")
        x = float(splitted_line[0])
        y = float(splitted_line[1])
        z = float(splitted_line[2])

        remaining_line = splitted_line[3:]
        landmark_name = " ".join(remaining_line)
        if landmark_name[-1:] == "\n":
            landmark_name = landmark_name[:-1]

        landmarks[landmark_name] = [x,y,z]

    return landmarks

def parse_landmark_txt_index_formatting(data):
    """
    Parse landamrk txt file with formatting
    landmark_index landmark_name

    :param data (List[str]) list of strings, each string
                represents one line from the txt file
    
    :return landmarks (dict) with formatting 
                      {landmark_name: index}
    """

    # get number of landmarks
    N = len(data)
    if data[-1] == "\n":
        N -= 1

    # define landmarks
    landmark_indices = {}

    for i in range(N):
        splitted_line = data[i].split(" ")
        ind = int(splitted_line[0])

        remaining_line = splitted_line[1:]
        landmark_name = " ".join(remaining_line)
        if landmark_name[-1:] == "\n":
            landmark_name = landmark_name[:-1]

        landmark_indices[landmark_name] = ind

    return landmark_indices

def load_landmarks(landmark_path: str, 
                   landmark_subset: List[str] = None, 
                   scan_vertices: np.ndarray = None):
    """
    Load landmarks from file and return the landmarks as
    torch tensor.

    Landmark file is defined in the following format:
    - .txt extension
        Option1) x y z landmark_name
        Option2) landmark_index landmark_name
    - .json extension
        Option1) {landmark_name: [x,y,z]}
        Option2) {landmark_name: landmark_index}
    - .lnd extension
        specific to the CAESAR dataset -> check landmarks.py
    where the landmark_index is the index of the landmark in 
    scan_vertices

    
    :param landmark_path: (str) of path to landmark file
    :param landmark_subset: (list) list of strings of landmark
                            names to use
    :param scan_vertices: (np.ndarray) dim (N,3) of the vertices
                          if landmarks defined as indices of the
                          vertices, returning landmarks as 
                          scan_vertices[landmark_indices,:]

                          
    Return: landmarks: (dict) of landmark_name: landmark_coords
                       where landmark_coords is list of 3 floats
    """

    # if empty landmark subset, return None
    if landmark_subset == []:
        return {}

    ext = landmark_path.split(".")[-1]
    supported_extensions = [".txt",".json",".lnd"]
    formatting_type = "indices"

    if ext == "txt":
        # read txt file
        with open(landmark_path, 'r') as file:
            data = file.readlines()

        # check formatting type
        try:
            _ = float(data[0].split(" ")[1])
            formatting_type = "coords"
        except Exception as e:
            pass

        # parse landmarks
        if formatting_type == "coords":
            landmarks = parse_landmark_txt_coords_formatting(data)
        elif formatting_type == "indices":
            if isinstance(scan_vertices,type(None)):
                msg = "Scan vertices need to be provided for"
                msg += "index type of landmark file formatting"
                raise NameError(msg)
            landmark_inds = parse_landmark_txt_index_formatting(data)
            landmarks = {}
            for lm_name, lm_ind in landmark_inds.items():
                landmarks[lm_name] = scan_vertices[lm_ind,:]

    elif ext == "json":
        with open(landmark_path,"r") as f:
            data = json.load(f)

        # check formatting type
        first_lm = list(data.keys())[0]
        if isinstance(data[first_lm],list):
            formatting_type = "coords"

        if formatting_type == "coords":
            landmarks = data
        elif formatting_type == "indices":
            if isinstance(scan_vertices,type(None)):
                msg = "Scan vertices need to be provided for"
                msg += "index type of landmark file formatting"
                raise NameError(msg)
            
            landmarks = {}
            for lm_name, lm_ind in data.items():
                landmarks[lm_name] = scan_vertices[lm_ind,:]

    elif ext == "lnd":
        print("Be aware that the .lnd extension assumes you are using the caesar dataset.")
        print("Automatically using scale of 1000 to scale the LM. Careful to not repeat the scaling.")
        landmarks = landmarks_module.process_caesar_landmarks(landmark_path,1000)

    else:
        supported_extensions_str = ', '.join(supported_extensions)
        msg = f"Landmark extensions supported: {supported_extensions_str}. Got .{ext}."
        raise ValueError(msg)

    # select subset of landmarks
    if not isinstance(landmark_subset,type(None)):
        landmarks_sub = {}
        for lm_name in landmark_subset:
            if lm_name in landmarks:
                landmarks_sub[lm_name] = landmarks[lm_name]

        landmarks =  landmarks_sub

    return landmarks

def load_scan(scan_path, return_vertex_colors=False):
    """
    Load scan given its scan_path using open3d. 
    Scan can be defined as:
    - .ply file
    - .ply.gz file


    :param scan_path: (str) of path to scan file
    """

    ext = scan_path.split(".")[-1]
    ext_extended = f"{scan_path.split('.')[-2]}.{ext}"
    supported_extensions = [".ply",".ply.gz", ".obj"]

    if ext in ["ply", "obj"]:
        scan = o3d.io.read_triangle_mesh(scan_path)
        scan_vertices = np.asarray(scan.vertices)
        scan_faces = np.asarray(scan.triangles)
        scan_faces = scan_faces if scan_faces.shape[0] > 0 else None
        if return_vertex_colors:
            scan_vertex_colors = np.asarray(scan.vertex_colors)
            print(scan_vertex_colors.shape)

    elif ext_extended == "ply.gz":
        with gzip.open(scan_path, 'rb') as gz_file:
            try:
                ply_content = gz_file.read()
            except Exception as _:
                raise ValueError("Cannot read .ply.gz file.")

            temp_ply_path = tempfile.mktemp(suffix=".ply")
            with open(temp_ply_path, 'wb') as temp_ply_file:
                temp_ply_file.write(ply_content)

            scan = o3d.io.read_triangle_mesh(temp_ply_path)
            scan_vertices = np.asarray(scan.vertices)
            scan_faces = np.asarray(scan.triangles)
            scan_faces = scan_faces if scan_faces.shape[0] > 0 else None
            if return_vertex_colors:
                scan_vertex_colors = np.asarray(scan.vertex_colors)
            os.remove(temp_ply_path)

    else:
        supported_extensions_str = ', '.join(supported_extensions)
        msg = f"Scan extensions supported: {supported_extensions_str}. Got .{ext}."
        raise ValueError(msg)

    if return_vertex_colors:
        return scan_vertices, scan_vertex_colors
    else:
        return scan_vertices, scan_faces

def load_fit(path):

    data = np.load(path)
    return data

#############################################################
#                   Random                                  #
#############################################################

def set_seed(sd):
    torch.manual_seed(sd)
    random.seed(sd)
    np.random.seed(sd)


#############################################################
#                   Point cloud                             #
#############################################################

def get_normals(vertices):
    """
    Find unit vertex normals.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    unit_normals = normals / np.linalg.norm(normals, axis=1)[:,None]
    return torch.from_numpy(unit_normals)

def update_normals(normals, A_mat):
    '''
    Rotates the normals by homogeneous transformation A (given as N x 3 x 4)
    A missing last row [0,0,0,1]

    Input:  normals: pytorch Tensor of N x 3 normals
            A: pytorch Tensor of N x 3 x 4 transformations (missing last row)
    Return: rot_normals: pytorch Tensor N x 3 of rotated normals A * normals
    '''
    
    N = normals.shape[0]
    device = normals.device
    
    # turn normals homo N x 4
    normals_homo = torch.cat([normals.float(),
                            torch.ones(N,device=device).unsqueeze(1).float()],1)
    
    # A is 3 x 4, want homogeneous transf, add last row [0,0,0,1]
    # A dim N x 4 x 4
    A_mat = torch.cat([A_mat,
                torch.tensor([0,0,0,1], device=device).unsqueeze(0).expand(N,1,4)],
                dim=1)
    
    # transform normals N x 4 x 4 * N x 4 x 1   =   N x 4 x 1
    transformed_normals = torch.matmul(A_mat, normals_homo.unsqueeze(2))
    # N x 4
    transformed_normals = transformed_normals.squeeze()
    # N x 3
    transformed_normals = torch.div(transformed_normals, 
                                transformed_normals[:,3].unsqueeze(1))[:,:3]    
    return  transformed_normals

def rotate_points_homo(points, A):
    '''
    input:  points: torch tensor N x 3
            A: torch tensor N x 3 x 4 
               (homogenous transf matrix without [0,0,0,1] last row)
    return: transformed_points: torch tensor N x 3
    '''
    N = points.shape[0]
    device = points.device
    
    # turn points homo N x 4
    points_homo = torch.cat([points.float(),
                             torch.ones(N, device=device).unsqueeze(1).float()],1)
    
    # A is 3 x 4, want homogeneous transf, add last row [0,0,0,1]
    # A dim N x 4 x 4
    A = torch.cat([A,
                   torch.tensor([0,0,0,1], device=device).unsqueeze(0).expand(N,1,4)],
                  dim=1)
    
    # transform points N x 4 x 4 * N x 4 x 1   =   N x 4 x 1
    transformed_points = torch.matmul(A, points_homo.unsqueeze(2))
    # N x 4
    transformed_points = transformed_points.squeeze()
    
    # remove homogeneous coord N x 3
    transformed_points = torch.div(transformed_points, 
                                   transformed_points[:,3].unsqueeze(1))[:,:3]    
    return  transformed_points# N x 3 



#############################################################
#                   Optimization                            #
#############################################################

def initialize_A(N, random_init=True):
        '''
        Creates (N,3,4) homogeneous transformation matrix
        Either random or eye matrix for rotation and 0 for translation.
        Homogeneous transf matrix without [0,0,0,1] to optimize space

        :param N: number of matrices to create
        :param random_init: boolean indicating if random initialization

        :return A: (N,3,4)torch tensor of homogeneous transformation matrices
        '''

        A = torch.cat([torch.diag(torch.ones(3)), 
                        torch.zeros(3,1)],dim=1).unsqueeze(0).expand(N,3,4)
        
        if random_init:
            # just random initialization
            # A_tnp = torch.rand(self.nr_A,3,4)
            # eye

            # displace the eye matrix a little
            r1 = -0.0001
            r2 = 0.0001
            A = A + torch.FloatTensor(N,3,4).uniform_(r1, r2)

        # else:
        #     A = torch.cat([torch.diag(torch.ones(3)), 
        #                     torch.zeros(3,1)],dim=1).unsqueeze(0).expand(N,3,4)
        return torch.tensor(A, requires_grad=True, device="cuda:0") #.requires_grad_(True)
        #return A.requires_grad_(True).cuda()
        # return A.clone().requires_grad_(True).cuda()


def exit_fitting_vertices(current_loss, previous_loss, loss_difference, minimal_loss):
    condition = (current_loss.item() <= minimal_loss) or \
                (torch.abs(current_loss-previous_loss)<=loss_difference)    
    
    return condition