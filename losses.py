
import torch
import torch.nn as nn
import sys
import numpy as np
import os
import pickle
from typing import List

import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path,"pyTorchChamferDistance"))
from chamfer_distance import ChamferDistance

# from utils import get_normals

LOSS_MAPPER = {"data":"DataLoss",
               "partial_data": "PartialDataLoss",
               "smooth":"SmoothnessLoss",
               "landmark":"LandmarkLoss",
               "normal":"NormalLoss",
               }

class DataLoss(nn.Module):
    def __init__(self,**kwargs):
        super(DataLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self,scan_vertices,template_vertices,**kwargs):
        '''
        Directional Chamfer Distance from bm template to vertices
        Sum the distances from every point of the template to the closest point
        of the scan

        :param scan_vertices: (torch.tensor) of N x 3 dim
        :param template_vertices: (torch.tensor) of M x 3 dim
        return: (float) sum of distances from every point of the 
                        template to the closest point of the scan
        '''
        return self.chamfer_dist(scan_vertices,template_vertices)[1].sum()
    

class PartialDataLoss(nn.Module):
    def __init__(self,partial_data_threshold: float,**kwargs):
        super(PartialDataLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()
        self.partial_data_threshold = partial_data_threshold

    def forward(self,scan_vertices,template_vertices,**kwargs):
        '''
        Directional Chamfer Distance from template to vertices
        Sum the distances from every point of the template to the closest point
        of the scan if the distance is lower than partial_data_threshold.

        :param scan_vertices: (torch.tensor) of N x 3 dim
        :param template_vertices: (torch.tensor) of M x 3 dim
        return: (float) sum of distances from every point of the 
                        template to the closest point of the scan closer than
                        partial_data_threshold
        '''
        _, template2scan_dist, _ , _ = self.chamfer_dist(scan_vertices,template_vertices)
        return template2scan_dist[template2scan_dist < self.partial_data_threshold].sum()


class SmoothnessLoss(nn.Module):
    def __init__(self,body_models_path: str,**kwargs):
        super(SmoothnessLoss, self).__init__()

        neighbor_pairs_path = os.path.join(body_models_path,
                                           "neighbor_pairs_indices.npy")
        all_neighbors = np.load(neighbor_pairs_path)
        self.all_neighbors = torch.from_numpy(all_neighbors).type(torch.long)

    def forward(self,A,**kwargs):
        '''
        Difference between homo transformations between neighboring template
        points

        :param A: (torch.tensor) transformation matrix A of N x 4 x 4 dim
        return: sum of Frobenious norms of the difference of each 
                neighborhood points (their transf matrices)
        '''
        diff = A[self.all_neighbors[:,0],:,:] - A[self.all_neighbors[:,1],:,:]
        return torch.sum(torch.abs(diff)**2)
    

class LandmarkLoss(nn.Module):
    def __init__(self, **kwargs):
        super(LandmarkLoss, self).__init__()

    def forward(self,scan_landmarks,template_landmarks,**kwargs):
        """
        summed L2 norm between scan_landmarks and template_landmarks

        :param scan_landmarks: (torch.tensor) dim (N,3)
        :param template_landmarks: (torch.tensor) dim (N,3)

        return: (float) summed L2 norm between scan_landmarks and 
                        template_landmarks
        """

        return torch.sum((scan_landmarks - template_landmarks)**2)
    

class NormalLoss(nn.Module):
    def __init__(self, 
                 normal_threshold_angle: float = None,
                 normal_threshold_distance: float = None):
        super(NormalLoss, self).__init__()
        self.normal_threshold_angle = normal_threshold_angle
        self.normal_threshold_distance = normal_threshold_distance

    # def get_scan_normals(self,scan):
    #     self.scan_normals = get_normals(scan)

    def forward(self, scan_vertices, template_vertices, 
                scan_normals, template_normals,
                K_knn=15, **kwargs):
        '''
        For each template vertex i, find the K_knn nearest neighbors in the scan
        and compute the angle between their normals and the normal at template vertex i.
        Choose the scan vertex with the smallest angle and compute the MSE between them.
        
        If using distance threshold, the template and scan vertices need to be closer 
        than normal_threshold_distance in order to include them in the loss. 
        If using angle threshold, the angle between the normals needs to be smaller 
        than normal_threshold_angle in order to include them in the loss.

        :param scan_vertices: (torch.tensor) dim 1 x N x 3
        :param template_vertices: (torch.tensor) dim 1 x M x 3
        :param scan_normals: (torch.tensor) dim N x 3
        :param template_normals: (torch.tensor) dim M x 3
        :param K_knn: (int) number of nearest neighbors to consider
        '''

        # N = scan_vertices.shape[0]
        if len(template_vertices.shape) == 3:
            template_vertices = template_vertices.squeeze()
        if len(scan_vertices.shape) == 3:
            scan_vertices = scan_vertices.squeeze()
        M = template_vertices.shape[0]

        scan_indices = []
        template_indices = []

        # 1. find knn for each template vertex
        for i in range(M):
            template_vert_i = template_vertices[i,:].unsqueeze(0) # (1,3)
            template_normal_i = template_normals[i,:].unsqueeze(0) # (1,3)

            # knn
            dist = torch.norm(scan_vertices - template_vert_i, dim=1) # (N)
            if self.normal_threshold_distance:
                if not any(dist < self.normal_threshold_distance):
                    continue
            scan_inds = torch.topk(dist,k=K_knn).indices # (K_knn)
            scan_normals_knn = scan_normals[scan_inds,:] # (K_knn,3)

            # FIXME: put this outside for loop?
            # find angle between template normal and scan normals
            dot_prod = torch.sum(torch.mul(scan_normals_knn,
                                           template_normal_i),dim=1)
            angle = torch.acos(torch.clamp(dot_prod,-1,1)) # (K_knn)
            angle_deg = torch.rad2deg(angle) # (K_knn)

            # if not isinstance(normal_threshold, type(None)):
            if self.normal_threshold_angle:
                # check if angle is below threshold
                if not any(angle_deg < self.normal_threshold_angle):
                    continue
            
            # get the smallest angle
            ind = torch.argmin(angle_deg).item()
            # save original index from scan
            scan_indices.append(scan_inds[ind])
            template_indices.append(i)

        scan_indices = torch.tensor(scan_indices)
        # 2. mse loss between the found points
        mse = torch.mean(torch.norm(scan_vertices[scan_indices,:] - 
                                   template_vertices[template_indices,:],dim=1))
        
        return mse
    

class Losses(nn.Module):
    """
    Loss class that combines multiple losses by weighting them and 
    summing them.
    
    """
    def __init__(self, cfg: dict, loss_weights: dict,**kwargs):
        super(Losses, self).__init__()
        
        self.loss_names = cfg["use_losses"]
        self.loss_tracker = LossTracker(self.loss_names)
        self.loss_weights = loss_weights
        self.current_loss_weights = loss_weights[0]

        self.loss_fns = {}
        for loss_name in self.loss_names:
            loss_fn = eval(LOSS_MAPPER[loss_name])(cfg) 
            self.loss_fns[loss_name] = loss_fn
    

    def track_loss(self,loss_dict):
        self.loss_tracker.update(loss_dict)

    def update_loss_weights(self,iteration):
        if iteration in self.loss_weights:
            self.current_loss_weights = self.loss_weights[iteration]

    def forward(self, **kwargs):


        loss = {loss_name: self.current_loss_weights[loss_name] * loss_fn(**kwargs) 
                    for loss_name, loss_fn in self.loss_fns.items()}
        self.track_loss(loss)
        loss = sum(loss.values())
        
        return loss
        

class LossTracker():

    def __init__(self,loss_names) -> None:
        self.losses = {name:[] for name in loss_names}
        self.losses["total"] = []

    def update(self,losses):
        for k,v in losses.items():
            self.losses[k].append(v.detach().cpu().item())


def summed_L2(x: torch.tensor, y: torch.tensor):
    """
    :param x: (torch.tensor) dim N x 3
    :param y: (torch.tensor) dim N x 3
    """
    return ((x-y)**2).sum(dim=1).sqrt().sum()


# PRIOR LOSS - FROM SMPLify paper
class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=torch.float32, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)