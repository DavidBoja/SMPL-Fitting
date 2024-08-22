
import torch
import smplx
import os

from landmarks import SMPL_INDEX_LANDMARKS

def infer_body_model(N_verts):
    if N_verts == 6890:
        return "smpl"
    else:
        raise NotImplementedError(f"Cannot infer the body model. \
                                  Body model with {N_verts} verts not implemented.")


class SMPLBodyModel():
    """
    Extended SMPL body model class used for optimization
    in order to deform the vertices with pose, shape, 
    scale and translation parameters.
    """
    
    def __init__(self, cfg: dict):
        
        self.all_landmark_indices = SMPL_INDEX_LANDMARKS
        self.gender = cfg["body_model_gender"].upper() if "body_model_gender" in cfg else "NEUTRAL"
        body_model_path = os.path.join(cfg["body_models_path"], 
                                       f"SMPL_{self.gender}.pkl")
        self.num_betas = cfg.get("body_model_num_betas", 10)
        self.body_model = smplx.create(body_model_path, 
                                        model_type="SMPL",
                                        gender=self.gender, 
                                        num_betas=self.num_betas,
                                        use_face_contour=False,
                                        ext='pkl')
        
        self.current_pose = None
        self.current_global_orient = None
        self.current_shape = None
        self.current_trans = None
        self.current_scale = None
        self.body_model_name = "smpl"

    @property
    def N_verts(self):
        return 6890

    @property
    def verts_t_pose(self):
        return self.body_model.v_template

    @property
    def verts(self):
        return self.body_model(body_pose=self.current_pose, 
                               betas=self.current_shape, 
                               global_orient=self.current_global_orient).vertices[0]

    @property
    def joints(self):
        return self.body_model(body_pose=self.current_pose, 
                               betas=self.current_shape, 
                               global_orient=self.current_global_orient).joints[0]

    @property
    def faces(self):
        return self.body_model.faces
    
    def landmark_indices(self,landmarks_order):
        return [self.all_landmark_indices[k] for k in landmarks_order]


    def cuda(self):
        self.body_model.cuda()

    def __call__(self, pose, betas, **kwargs):

        self.current_pose = pose[:,3:]
        self.current_global_orient = pose[:,:3]
        self.current_shape = betas

        body_pose = pose[:,3:]
        global_orient = pose[:,:3]
        return self.body_model(body_pose=body_pose, 
                               betas=betas, 
                               global_orient=global_orient)
    
    def deform_verts(self, 
                     pose: torch.tensor,
                     betas: torch.tensor,
                     trans: torch.tensor,
                     scale: torch.tensor):
        
        self.current_pose = pose[:,3:]
        self.current_global_orient = pose[:,:3]
        self.current_shape = betas
        self.current_trans = trans
        self.current_scale = scale

        body_pose = pose[:,3:]
        global_orient = pose[:,:3]
        deformed_verts = self.body_model(body_pose=body_pose, 
                                         betas=betas, 
                                         global_orient=global_orient).vertices[0]

        # return (deformed_verts + trans) * scale
        return (deformed_verts * scale) + trans


class BodyModel():
    """
    Class used to optimize parameters of the
    SMPL / SMPLX body models.
    """

    def __new__(cls, cfg):

        possible_model_types = ["smpl"] #["smpl", "smplx"]
        model_type = cfg["body_model"].lower()

        if model_type == "smpl":
            return SMPLBodyModel(cfg)
        # elif model_type == "smplx":
        #     return SMPLXBodyModel()
        else:
            msg = f"Model type {model_type} not defined. \
                    Possible model types are: {possible_model_types}"
            raise NotImplementedError(msg)