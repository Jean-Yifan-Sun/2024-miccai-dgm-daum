"""
DownstreamEvaluator.py

Run Downstream Tasks after training has finished
"""
import os
import torch
import logging


class DownstreamEvaluator(object):
    """
    Downstream Tasks
        - run tasks at training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """

    def __init__(self, name, evaluation_mode, model, device, test_data_dict, checkpoint_path):
        """
        @param model: nn.Module
            the neural network module
        @param device: torch.device
            cuda or cpu
        @param test_data_dict:  dict(datasetname, datasetloader)
            dictionary with dataset names and loaders
        @param checkpoint_path: str
            path to save results
        """
        self.name = name
        self.evaluation_mode = evaluation_mode
        self.evaluation_strategy = evaluation_mode.get_strategy()
        self.model = model.to(device)
        self.device = device
        self.test_data_dict = test_data_dict
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.image_path = checkpoint_path + '/images/'
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        super(DownstreamEvaluator, self).__init__()

        self.barrier()
        self.evaluation_strategy.prepare_evaluator(self)

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
            dictionary with the model weights of the federated collaborators
        """
        raise NotImplementedError("[DownstreamEvaluator::start_task]: Please Implement start_task() method")

    def gather_into_tensor(self, tensor: torch.Tensor):
        return self.evaluation_strategy.gather_into_tensor(tensor)

    @property
    def actual_model(self):
        return self.evaluation_strategy.get_actual_model(self)

    def log_to_wandb(self, *args, **kwargs):
        self.evaluation_strategy.log_wandb(*args, **kwargs)

    def barrier(self, name=""):
        logging.info(f"[Trainer::barrier]: Reached barrier {name}. Waiting...")
        self.evaluation_strategy.barrier()
        logging.info(f"[Trainer::barrier]: Barrier {name} passed.")