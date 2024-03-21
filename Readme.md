# SMPL-Fitting

Fit an SMPL body model (BM) to a given scan and view the optimization process in a plotly dashboard.  Fitting supported:
-  🧍‍♂️ fit the body model parameters (shape, pose, translation, scale)
-  🤹 fit the vertices to the scan

The code supports fitting a single scan 👤 or a whole dataset 👥.

<br>

https://github.com/DavidBoja/SMPL-Fitting/assets/32020857/34f3b1ce-81b8-460c-b645-fa04558bb1af



<br>

## 🔨 Getting started
You can use a docker container to facilitate running the code. After cloning the repo, run in terminal:

```bash
cd docker
sh build.sh
sh docker_run.sh CODE_PATH
```

by adjusting the `CODE_PATH` to the `SMPL-Fitting` directory location. This creates a `smpl-fitting-container` container. You can attach to it by running:

```bash
docker exec -it smpl-fitting-container /bin/bash
```

🚧 If you do not want to use docker, you can install the `docker/requirements.txt` into  your own environment. 🚧

Next, initialize the chamfer distance submodule by running:

```bash
git submodule update --init --recursive
```
<br>

Necessary files:
- put the `SMPL_{GENDER}.pkl` (MALE, FEMALE and NEUTRAL) models into the `data/body_models/smpl` folder. You can obtain the files [here](https://github.com/vchoutas/smplx#downloading-the-model).
- put the `gmm_08.pkl` prior into the `data/prior` folder. You can obtain the files [here](https://smplify.is.tuebingen.mpg.de/index.html).
- [OPTIONAL] We provide a demo for fitting the whole FAUST dataset. To do that, download the FAUST dataset [here](https://faust-leaderboard.is.tuebingen.mpg.de/download) and put the `FAUST/training/scans` and `FAUST/training/registrations` folders into the `data/FAUST/training` folder in this repository. We already provided the landmarks for fitting in `data/FAUST/training/landmarks`. 


<br>

## 🏃‍♀️ Fitting

The configuration files for the fitting are stored in the `configs` folder:
- `config.yaml` stores general variables and optimization-specific variables
- `loss_weight_configs.yaml` stores the loss weight strategy for the fitting process defined as `iteration: dict of loss weights` pairs. For example, `4: {"data": 1, "smooth": 150, "landmark": 100}` means that at iteration 4, the data loss will be multiplied by 1, smoothnesss loss will be multiplied by 150, etc.

<br>

### ⚙️ General configurations

The general configuration variables are listed below. The specific variables for fitting the body model are given [here](###-🧍‍♂️-Fit-body-model-(BM)), and the specific variables for fitting the vertices are given [here](###-🤹-Fit-vertices).

General variables:
- `verbose` - (bool) printout losses and variable values at each step
- `default_dtype` -  (torch.dtype) define the shape, pose, etc. tensor data types
- `pause_script_after_fitting` - (bool) pause the script after the fitting is done so you can visualize in peace
- `experiment_name` - (string) name your experiment

Visualization variables:
- `socket_type` - (string) type of socket, only zmq supported
- `socket_port` - (int) port for visualizations, localhost:socket_port
- `error_curves_logscale` - (bool) visualize loss curves in log scale
- `visualize` - (bool) visualize or not the fitting
- `visualize_steps` - (list / range) iterations to visualize, can be defined as summed ranges and lists, ex. `range(0, 500, 50)+[10,30,499]`

Path variables:
- `body_models_path` - (string) path to SMPL,SMPLX,.. body models
- `prior_path` - (string) path to the gmm prior loss .pkl file
- `save_path` - (string) path to save the results

Dataset variables for FAUST (these are dataset specific for each dataset you implement):
- `data_dir` - (string) path to FAUST dataset
- `load_gt` - (bool) load ground truth SMPL fitting or not

<br>

### 🧍‍♂️ Fit body model (BM)

Optimize the body model parameters of shape and pose (including translation and scale) that best fit the given scan. Check [notes on losses](##-📝-Notes) to see the losses used.

The optimization-specific configurations to fit a BM to a scan are set under `fit_body_model_optimization` in `config.yaml` with the following variables:

- `iterations` - (int) number of iterations
- `lr` - (float) learning rate
- `start_lr_decay_iteration` - (int) iteration when to start the learning rate decay calculated as `lr *(iterations-current iteration)/iterations`
- `body_model` - (string) which BM to use (smpl, smplx,..). See [Notes](##-📝-Notes) for supported models
- `use_landmarks` - (string / list) which body landmarks to use for fitting. Can be `All` to use all possible landmarks, `{BM}_INDEX_LANDMARKS` defined in landmarks.py or list of landmark names e.g. `["Lt. 10th Rib", "Lt. Dactylion",..]` defined in landmarks.py
- `loss_weight_option` - the strategy for the loss weights, defined in `loss_weights_configs.yaml` under `fit_bm_loss_weight_strategy`

The default variables already set should work well for the fitting process.

#### 👤 Fit BM to single scan

```bash
python fit_body_model.py onto_scan --scan_path {path-to-scan} --landmark_path {path-to-landmarks}
```

Check [Notes](##-📝-Notes) to see the supported scan and landmark file extensions.

#### 👥 Fit BM to dataset

```bash
python fit_body_model.py onto_dataset --dataset_name {dataset-to-fit}
```

The dataset you want to fit needs to be defined in `datasets.py` as a torch dataset. Check [notes on datasets](##-📝-Notes) for more details. We already provide the FAUST dataset in `datasets.py`.

<br>



### 🤹 Fit vertices

Optimize the vertices of a BM (or mesh) that best fit the given scan. Check [notes on losses](##-📝-Notes) to see the losses used.

The optimization-specific configuration to fit the vertices to a scan is set under `fit_vertices_optimization` in `config.yaml` with the following variables:

- `max_iterations` - (int) number of maximal iterations
- `stop_at_loss_value` - (float) stop fitting if loss under this threshold
- `stop_at_loss_difference` - (float) stop fitting if difference of loss at iteration `i-1` and iteration `i` is less this threshold
- `use_landmarks` - (string / list) which body landmarks to use for fitting. Can be `nul` to not use landmarks, `All` to use all possible landmarks, `{BM}_INDEX_LANDMARKS` defined in landmarks.py, or list of landmark names e.g. `["Lt. 10th Rib", "Lt. Dactylion",..]` defined in landmarks.py
- `random_init_A` - (bool) random initialization of vertices transformation
- `seed` - (float) seed for random initialization of vertices transformation
- `use_losses` - (list) losses to use. The complete list of losses is `["data","smooth","landmark","normal","partial_data"]`. Check [notes on losses](##-📝-Notes).
- `loss_weight_option` - (string) the strategy for the loss weights, defined in `loss_weights_configs.yaml` under `fit_verts_loss_weight_strategy`
- `lr` - (float) learning rate
- `normal_threshold_angle` - (float) used if normal loss included in `use_losses`. Penalizes knn points only if angle is lower than this threshold. Otherwise points are ignored
- `normal_threshold_distance` - (float) used if normal loss included in `use_losses`. Penalizes knn points only if the distance is lower than this threshold. Otherwise points are ignored
- `partial_data_threshold` - (float) used if partial_data loss included in `use_losses`. Chamfer distance from BM to scan for points that are closer than this threshold. Otherwise points are ignored

<br>



#### 👤 Fit vertices to scan

```bash
python fit_vertices.py onto_scan --scan_path {path-to-scan} --landmark_path {path-to-landmarks} --start_from_previous_results {path-to-YYYY_MM_DD_HH_MM_SS-folder}
```

Check [Notes](##-📝-Notes) to see the supported scan and landmark file extensions. You can either use `--start_from_previous_results` to fit the vertices of the previously fitted BM with the `fit_body_model.py` script ( ⚠️ provide the folder where the fitted `.npz` is located) or use `--start_from_body_model` to start fitting a BM with zero shape and pose to the scan (⚠️ results will probably be poor). 



#### 👥 Fit vertices to dataset

```bash
python fit_vertices.py onto_dataset --dataset_name {dataset-name} --start_from_previous_results {path-to-previously-fitted-bm-results}
```
You can either use `--start_from_previous_results` to fit the vertices of the previously fitted BM with the `fit_body_model.py` script (⚠️ provide the folder where the fitted `.npz` are located) or use `--start_from_body_model` to start fitting a BM with zero shape and pose to the scan (⚠️ results will be poor).
The dataset you want to fit needs to be defined in `datasets.py` as a torch dataset. Check [notes on datasets](##-📝-Notes) for more details. We already provide the FAUST dataset in `datasets.py`.

<br>
<br>


## ⚖️ Evaluate
Use the `evaluate_fitting.py` script to evaluate the fitting.

<br>

### evaluate per-vertex-error
Evaluate the per vertex error (pve) which is the average euclidean distance between the given ground truth BM to the fitted BM.

```python
python evaluate_fitting.py pve -F {path-to-results}
```

The pve unit is determined by the data. For the FAUST dataset the unit is given in meters.

You can use:
- `-V` -  to visualize the pve for each example
- `--select_examples` -  (list) to select a subset of examples to evaluate (only if evaluating fitting to dataset)
- `--ground_truth_path` - (string) to set the path to the ground truth body model (only if evaluating fitting to scan)

<br>

### evaluate chamfer distance
Evaluate the (various definitions of) chamfer distance (CD) from the estimated body model to the scan with:

```python
python evaluate_fitting.py chamfer -F {path-to-results}
```

where the different definitions are:

- `Chamfer standard` is (mean(dists_bm2scan) + mean(dists_scan2bm))
- `Chamfer bidirectional` is mean(concatenation(dists_bm2scan,dists_scan2bm))
- `Chamfer from body model to scan` is mean(dists_bm2scan)
- `Chamfer from scan to body model` is mean(dists_scan2bm)

and are averaged over the examples. The unit of these metrics is determined by the data. For the FAUST dataset the unit is given in meters.

You can use:
- `--select_examples` to select a subset of examples to evaluate (only if evaluating fitting to dataset)
- `--device` to set gpu for running a faster chamfer distance (use `cuda:{gpu-index}`)
- `--scan_path` - (string) to set the path to the scan you are evaluating (only if evaluating fitting to scan and not whole dataset)

<br>

## 📈 Visualization
1. Visualize SMPL landmarks with:

    ```bash
    python visualization.py visualize_smpl_landmarks
    ```
2. Visualize scan landmarks with:
    ```bash
    python visualization.py visualize_scan_landmarks --scan_path {path-to-scan} --landmark_path {path-to-landmarks}
    ```
    Check [Notes](##-📝-Notes) section to find out the possible landmark definitions.
3. Visualize fitting:
    ```bash
    python visualization.py visualize_fitting --scan_path {path-to-scan} --fitted_npz_file {path-to-.npz-file}
    ```
    where the `.npz` is obtained with the fitting scripts.


<br>

## 📝 Notes

### Notes on landmarks
The list of available landmarks for each BM are listed in `landmarks.py`. \
The supported ways of loading landmarks for a scan are:

- `.txt` extension has two options
  1. `x y z landmark_name`
  2. `landmark_index landmark_name`
- `.json` extension has two options
  1. `{landmark_name: [x,y,z]}`
  2. `{landmark_name: landmark_index}`

where `x y z` indicate the coordinates of the landmark and `landmark_index` indicates the index of the scan vertex representing the landmark.

<br>

### Notes on losses

Losses for fitting the BM:

- `data loss` - chamfer distance between BM and scan
- `landmark loss` - L2 distance between BM landmarks and scan landmarks
- `prior shape loss` - L2 norm of BM shape parameters
- `prior pose loss` - gmm prior loss from [1]

Losses for fitting the vertices:

- `data loss` - directional chamfer distance from BM to scan
- `smoothness loss` - difference between transformations of neighboring BM vertices
- `landmark loss` - L2 distance between BM landmarks and scan landmarks
- `normal loss` - L2 distance between points with normals within angle threshold

<br>

### Notes on datasets

The dataset you want to fit needs to be defined in `datasets.py` as a torch dataset with the following variables:

- `name` - (string) name of the scan
- `vertices` - (np.ndarray) vertices of the scan
- `faces` - (np.ndarray) faces of the scan (set to `None` if no faces)
- `landmarks` - (dict) of (landmark_name: landmark_coords) pairs where landmark_coords is list of 3 floats

If you additionally want to evaluate the per vertex error (pve) after fitting (check [⚖️ Evaluate](##-⚖️-Evaluate)) which compares the mean L2 between the fitted BM and the ground truth BM, you need to provide the ground truth BM as:

- `vertices_gt` - (np.ndarray) ground truth vertices of the BM
- `faces_gt` - (np.ndarray) ground truth faces of the BM

We provide the FAUST and CAESAR dataset implementations in `datasets.py`. You can obtain the datasets from [here](https://faust-leaderboard.is.tuebingen.mpg.de/) and [here](https://bodysizeshape.com/page-1855750).

<br>

### Notes on supported BM

Currently, we support the SMPL body model.
If you want to add another BM, you can follow these steps:
1. Add the body models into `data/body_models`
2. Implement the body model in `body_models.py`
3. Implement the body model parameters in `body_parameters.py`
4. Implement the body landmarks in `landmarks.py`

<br>

## 💿 Demos

Fit body model onto scan:
```bash
python fit_body_model.py onto_scan --scan_path data/demo/tr_scan_000.ply --landmark_path data/demo/tr_scan_000_landmarks.json
```

Fit body model onto dataset (🚧 you need to provide the FAUST dataset files as mentioned above 🚧):
```bash
python fit_body_model.py onto_dataset -D FAUST
```

Fit the vertices of the previously fitted BM onto the scan even further:
```bash
python fit_vertices.py onto_scan --scan_path data/FAUST/training/scans/tr_scan_000.ply --landmark_path data/FAUST/training/landmarks/tr_scan_000_landmarks.json --start_from_previous_results data/demo
```

Fit the vertices of the previously fitted BM onto FAUST dataset further:
```bash
python fit_vertices.py onto_dataset --dataset_name FAUST --start_from_previous_results data/demo
```

🚧 We provide only the fitted paths for scans tr_scan_000 and tr_scan_001. Therefore the rest of the scans are going to be skipped 🚧

Evaluate PVE of fitted scan for the two provided fittings:
```bash
python evaluate_fitting.py pve -F data/demo -G data/demo
```

Evaluate chamfer of fitted scan for the two provided fittings:
```bash
python evaluate_fitting.py chamfer -F data/demo
```

Visualize SMPL landmarks:
```bash
python visualization.py visualize_smpl_landmarks
```

Visualize FAUST scan landmarks:
```bash
python visualization.py visualize_scan_landmarks --scan_path data/demo/tr_scan_000.ply --landmark_path data/demo/tr_scan_000_landmarks.json
```

Visualize the fitted vertices of the BM onto the FAUST scan:
```bash
python visualization.py visualize_fitting --scan_path data/demo/tr_scan_000.ply --fitted_npz_file data/demo/tr_scan_000.npz
```

<br>

## Citation

Please cite our work and leave a star ⭐ if you find the repository useful.

```bibtex
@misc{SMPL-Fitting,
  author = {Bojani\'{c}, D.},
  title = {SMPL-Fitting},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavidBoja/SMPL-Fitting}},
}
```

## Todo
- [ ] Implement SMPLx body model


<br>

## References 
[1] Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image
