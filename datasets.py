
import torch.utils.data as data
from typing import List
import os
from glob import glob
import open3d as o3d
import gzip
from landmarks import process_caesar_landmarks
from utils import load_landmarks
import open3d as o3d
import numpy as np
import tempfile
import pandas as pd
    

class CAESAR(data.Dataset):
    '''
    CAESAR dataset
    z-ax is the height
    Returns vertices and landamrks in m, measurements in mm
    '''
    def __init__(self, 
                 data_dir: str,
                 load_countries: str = "All",
                 landmark_subset: str = None,
                 load_measurements: bool = False,
                 load_only_standing_pose: bool = False,
                 load_only_sitting_pose: bool = False,
                 **kwargs): 
        
        """
        :param data_dir (str): path to caesar dataset
        :param load_countries (str or list): countries to load. 
                                                If "All", all countries are loaded
        :param use_landmarks (str or list): landmarks to use. If "All", all 
                                            landmarks are loaded, if list of landmark names, 
                                            only those are loaded
        :param load_measurements (bool): whether to load measurements or not
        :param only_standing_pose (bool): load only standing pose from CAESAR
        :param only_sitting_pose (bool): load only sitting pose from CAESAR
        """
        self.landmark_subset = landmark_subset
        self.load_measurements = load_measurements
        self.load_only_standing_pose = load_only_standing_pose
        self.load_only_sitting_pose = load_only_sitting_pose
        
        # set loading countries
        all_countries = ["Italy","The Netherlands","North America"]
        if load_countries == "All":
            load_countries = all_countries

        for country in load_countries:
            if country not in all_countries:
                msg = f"Country {country} not found. Available countries are: {all_countries}"
                raise ValueError(msg)

        # set paths
        scans_and_landmark_dir = os.path.join(data_dir, "Data AE2000")
        scans_and_landmark_paths = {
            "Italy": os.path.join(scans_and_landmark_dir, "Italy","PLY and LND Italy"),
            "The Netherlands": os.path.join(scans_and_landmark_dir, "The Netherlands","PLY and LND TN"),
            "North America": os.path.join(scans_and_landmark_dir, "North America","PLY and LND NA")
            }
        self.scans_and_landmark_paths = scans_and_landmark_paths

        if self.load_measurements:
            measurements_path = os.path.join(data_dir, 
                                            "processed_data", 
                                            "measurements.csv")
            self.measurements = pd.read_csv(measurements_path)
            measurements_extr_seat_path = os.path.join(data_dir, 
                                                    "processed_data", 
                                                    "measurements_extracted_seated.csv")
            self.meas_extr_seated = pd.read_csv(measurements_extr_seat_path)
            measurements_extr_stand_path = os.path.join(data_dir, 
                                            "processed_data", 
                                            "measurements_extracted_standing.csv")
            self.meas_extr_stand = pd.read_csv(measurements_extr_stand_path)
            demographics_path = os.path.join(data_dir, 
                                            "processed_data", 
                                            "demographics.csv")
            self.demographics = pd.read_csv(demographics_path)

        self.scan_paths = []
        self.landmark_paths = []
        self.countries = []

        for country, path in scans_and_landmark_paths.items():
            for scan_path in glob(f"{path}/*.ply.gz"):

                scan_pose = scan_path.split("/")[-1].split(".ply.gz")[0][-1]

                if self.load_only_standing_pose:
                    if scan_pose != "a":
                        continue

                if self.load_only_sitting_pose:
                    if scan_pose != "b":
                        continue

                # set scan path
                self.scan_paths.append(scan_path)

                # set landmark path
                landmark_path = scan_path.replace(".ply.gz", ".lnd")
                if os.path.exists(landmark_path):
                    self.landmark_paths.append(landmark_path)
                else:
                    self.landmark_paths.append(None)

                # set country 
                self.countries.append(country)


            
        self.dataset_size = len(self.scan_paths)
        self.LANDMARK_SCALE = 1000 # landmark coordinates are in mm, we want them in m
        self.country_scales = {"Italy": 1, "The Netherlands": 1000, "North America": 1} # scale to convert from mm to m
        

    def __getitem__(self, index):
        """
        :return (dict): dictionary with keys:
            "name": name of scan
            "vertices": (N,3) np.array
            "faces": (N,3) np.array or None if no faces
            "landmarks": dict with landmark names as keys and landmark coords as values
                         landmark coords are (1,3) np.array or None if no landmarks
            "country": (string)
            "measurements": dict with measurements
            "demographics": dict with demographics
            "measurements_seat": dict with measurements in seated pose
            "measurements_stand": dict with measurements in standing pose
        """

        # load country
        scan_country = self.countries[index]
        scan_scale = self.country_scales[scan_country]

        # load scan
        scan_path = self.scan_paths[index]
        scan_name = os.path.basename(scan_path).split(".ply.gz")[0]
        scan_number = int(scan_name[-5:-1])

        with gzip.open(scan_path, 'rb') as gz_file:
            try:
                ply_content = gz_file.read()
            except Exception as _:
                return {"name": scan_name,
                        "vertices": None,
                        "faces": None,
                        "landmarks": None,
                        "country": None,
                        "measurements": None,
                        # "demographics": None
                        }


            # OPEN3D APPROACH
            temp_ply_path = tempfile.mktemp(suffix=".ply")
            with open(temp_ply_path, 'wb') as temp_ply_file:
                temp_ply_file.write(ply_content)

            scan = o3d.io.read_triangle_mesh(temp_ply_path)
            # scan_center = scan.get_center()
            scan_vertices = np.asarray(scan.vertices) / scan_scale
            scan_faces = np.asarray(scan.triangles)
            scan_faces = scan_faces if scan_faces.shape[0] > 0 else None
            os.remove(temp_ply_path)

        # load landmarks
        landmark_path = self.landmark_paths[index]
        if landmark_path is not None:
            landmarks = process_caesar_landmarks(landmark_path, 
                                                 self.LANDMARK_SCALE)
                
            if isinstance(self.landmark_subset, list):
                landmarks = {lm_name: landmarks[lm_name] 
                             for lm_name in self.landmark_subset 
                             if lm_name in landmarks.keys()}
        else:
            landmarks = None

        # load measurements
        if self.load_measurements:
            measurements = self.measurements.loc[
                    (self.measurements["Country"] == scan_country) & 
                    (self.measurements["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements = None if measurements == [] else measurements[0]
            
            measurements_seat = self.meas_extr_seated.loc[
                    (self.meas_extr_seated["Country"] == scan_country) & 
                    (self.meas_extr_seated["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements_seat = None if measurements_seat == [] else measurements_seat[0]
            
            measurements_stand = self.meas_extr_stand.loc[
                    (self.meas_extr_stand["Country"] == scan_country) & 
                    (self.meas_extr_stand["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements_stand = None if measurements_stand == [] else measurements_stand[0]
            
            demographics = self.demographics.loc[
                    (self.demographics["Country"] == scan_country) & 
                    (self.demographics["Subject Number"] == scan_number)
                    ].to_dict("records")
            demographics = None if demographics == [] else demographics[0]
        else:
            measurements = None
            measurements_seat = None
            measurements_stand = None
            demographics = None
        

        return {"name": scan_name,
                "vertices": scan_vertices,
                "faces": scan_faces,
                "landmarks": landmarks,
                "country": self.countries[index],
                "measurements": measurements,
                "measurements_seat": measurements_seat,
                "measurements_stand": measurements_stand,
                "demographics": demographics
                }

    def __len__(self):
        return self.dataset_size


class FAUST(data.Dataset):
    '''
    FAUST dataset
    y-ax is the height
    '''
    def __init__(self,
                 data_dir: str,
                 load_gt: bool = True,
                 landmark_subset: str = None,
                 **kwargs): 
        
        """
        :param data_dir (str): path to caesar dataset
        :param load_gt (bool): whether to load ground truth fitting or not
                                using registrations folder from FAUST as ground truth
        :param use_landmarks (str or list): landmarks to use. If "All", all 
                                            landmarks are loaded, if list of landmark names, 
                                            only those are loaded, if string, then see landmarks.py
                                            for the definition

        """
        self.load_gt = load_gt
        self.landmark_subset = landmark_subset
        self.gt_available = True
        
        self.gender = ["Male"] * 10 + \
                      ["Female"] * 10 + \
                      ["Male"] * 20 + \
                      ["Female"] * 30 + \
                      ["Male"] * 10 + \
                      ["Female"] * 10 + \
                      ["Male"] * 10

        scans_dir_path = os.path.join(data_dir, "scans")
        registrations_dir_path = os.path.join(data_dir, "registrations")
        landmarks_dir_path = os.path.join(data_dir, "landmarks")

        self.scan_paths = []
        self.registration_paths = []
        self.landmark_paths = []

        for i in range(100):
            index = str(i).zfill(3)

            scan_name = f"tr_scan_{index}.ply"
            scan_path = os.path.join(scans_dir_path, scan_name)
            self.scan_paths.append(scan_path)

            registration_name = f"tr_reg_{index}.ply"
            registration_path = os.path.join(registrations_dir_path, registration_name)
            self.registration_paths.append(registration_path)

            landmark_name = f"tr_scan_{index}_landmarks.json"
            landmark_path = os.path.join(landmarks_dir_path, landmark_name)
            self.landmark_paths.append(landmark_path)

            
        self.dataset_size = len(self.scan_paths)
        

    def __getitem__(self, index):
        """
        :return (dict): dictionary with keys:
            "name": name of scan
            "vertices": (N,3) np.array
            "faces": (N,3) np.array or None if no faces
            "landmarks": dict with landmark names as keys and landmark coords as values
                         landmark coords are (1,3) np.array or None if no landmarks
            "vertices_gt": (N,3) np.array of ground truth fitted body model
            "faces_gt": (N,3) np.array of ground truth fitted body model or None if no faces
        """

        # load scan
        scan_path = self.scan_paths[index]
        scan_name = os.path.basename(scan_path).split(".ply")[0]
        scan = o3d.io.read_triangle_mesh(scan_path)
        scan_vertices = np.asarray(scan.vertices)
        scan_faces = np.asarray(scan.triangles)
        scan_faces = scan_faces if scan_faces.shape[0] > 0 else None

        # load landmarks
        landmark_path = self.landmark_paths[index]
        landmarks = load_landmarks(landmark_path,
                                   self.landmark_subset,
                                   scan_vertices)

        return_dict = {"name": scan_name,
                        "vertices": scan_vertices,
                        "faces": scan_faces,
                        "landmarks": landmarks,
                        }
        
        # load ground truth
        if self.load_gt:
            registration_path = self.registration_paths[index]
            registration = o3d.io.read_triangle_mesh(registration_path)
            registration_vertices = np.asarray(registration.vertices)
            registration_faces = np.asarray(registration.triangles)
            registration_faces = registration_faces if registration_faces.shape[0] > 0 else None

            registration_dict = {"vertices_gt": registration_vertices,
                                "faces_gt": registration_faces}

            return_dict.update(registration_dict)
        
        return return_dict

    def __len__(self):
        return self.dataset_size