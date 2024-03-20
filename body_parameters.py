

import torch



class OptimizationSMPL(torch.nn.Module):
    """
    Class used to optimize SMPL parameters.
    """
    def __init__(self, cfg: dict):
        super(OptimizationSMPL, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 72).cuda())
        # self.pose = torch.nn.Parameter(torch.zeros(1, 69).cuda())
        self.beta = torch.nn.Parameter((torch.zeros(1, 10).cuda()))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())
        self.scale = torch.nn.Parameter(torch.ones(1).cuda()*1)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale
    


class BodyParameters():

    def __new__(cls, cfg):

        possible_model_types = ["smpl"] #["smpl", "smplx"]
        model_type = cfg["body_model"].lower()

        if model_type == "smpl":
            return OptimizationSMPL(cfg)
        # elif model_type == "smplx":
        #     return OptimizationSMPLX()
        else:
            msg = f"Model type {model_type} not defined. \
                    Possible model types are: {possible_model_types}"
            raise NotImplementedError(msg)
        

