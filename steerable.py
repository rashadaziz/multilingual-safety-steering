import torch as t

class SteerableModel:
    """
    This class wraps PyTorch API-compatible models to perform interventions on its hidden states
    """

    interventions: dict[int, t.utils.hooks.RemovableHandle]

    def __init__(self, model: t.nn.Module):
        
        pass