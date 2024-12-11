"""
Trainer.py

Default class for running training

"""
from enum import Enum

import wandb
import copy

from core.TrainingMode import *
from dl_utils import *
import torch
from torch.nn import MSELoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from core.Main import is_master, get_rank
from optim.losses import PerceptualLoss
import os


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=25, min_delta=10e-9):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        print(f"INFO: Early stopping delta {min_delta}")
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            return False
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience} with {self.best_loss - val_loss}")

            if self.counter >= self.patience:
                self.counter = 0
                print('INFO: Early stopping')
                return True


class Trainer:
    def __init__(self, training_params, training_mode, model, data, device, log_wandb=True):
        """
        Init function for Client
        :param training_params: list
            parameters for local training routine
        :param device: torch.device
            GPU  |  CPU
        :param model: torch.nn.module
            Neural network
        """
        self.training_mode = training_mode
        self.training_strategy = training_mode.get_strategy()
        self.barrier("trainer init")

        if 'checkpoint_path' in training_params:
            self.client_path = training_params['checkpoint_path']
            if not os.path.exists(self.client_path):
                try:
                    os.makedirs(self.client_path)
                except FileExistsError:
                    pass

        self.training_params = training_params

        self.train_ds, self.val_ds, self.test_ds = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
        # logging.info(f'train loader: {len(self.train_ds)}, val loader: {self.val_ds}, test loader: {self.test_ds}')
        self.num_train_samples = len(self.train_ds) * self.train_ds.batch_size

        self.device = device
        self.model = model.train().to(self.device)
        self.test_model = copy.deepcopy(model.eval().to(self.device))
        self.barrier("model_init")

        patience = training_params['patience'] if 'patience' in training_params.keys() else 25
        self.early_stopping = EarlyStopping(patience=patience)

        self.log_wandb = log_wandb

        if log_wandb:
            wandb.watch(self.model)
        self.barrier("wandb")
        # Optimizer
        opt_params = training_params['optimizer_params']
        self.optimizer = Adam(self.model.parameters(), **opt_params)

        self.lr_scheduler = None
        lr_sch_type = training_params['lr_scheduler'] if 'lr_scheduler' in training_params.keys() else 'none'

        if lr_sch_type == 'cosine':
            self.optimizer = Adam(self.model.parameters(), lr=training_params['optimizer_params']['lr'],
                                  amsgrad=True, weight_decay=0.00001)
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100)
        elif lr_sch_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_sch_type == 'exponential':
            self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.97)

        loss_class = import_module(training_params['loss']['module_name'],
                                   training_params['loss']['class_name'])
        self.criterion_rec = loss_class(**(training_params['loss']['params'])) \
            if training_params['loss']['params'] is not None else loss_class()

        if 'transformer' not in training_params.keys():
            self.transform = None
        else:
            transform_class = import_module(training_params['transformer']['module_name'],
                                            training_params['transformer']['class_name']) \
                if 'module_name' in training_params['transformer'].keys() else None

            self.transform = transform_class(**(training_params['transformer']['params'])) \
                if transform_class is not None else None

        self.criterion_MSE = MSELoss().to(device)
        self.criterion_PL = PerceptualLoss(device=device)
        self.min_val_loss = np.inf
        self.alfa = training_params['alfa'] if 'alfa' in training_params.keys() else 0

        self.best_weights = self.model.state_dict()
        self.best_opt_weights = self.optimizer.state_dict()

        self.barrier()
        self.training_strategy.prepare_trainer(self)

    def get_nr_train_samples(self):
            return self.num_train_samples

    def train(self, model_state=None, opt_state=None, accountant=None, epoch=0):
        """
        Train local client
        :param w_global: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :return:
            self.model.state_dict():
        """
        # return self.model.state_dict()
        raise NotImplementedError("[Trainer::train]: Please Implement train() method")

    def test(self, model_weights, test_data, task='Val', optimizer_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return:
            metrics: dict
                Dictionary with metrics:
                metric_name : value
                e.g.:
                metrics = {
                    'test_loss_l1': 0,
                    'test_loss_gdl': 0,
                    'test_total': 0
                }
            num_samples: int
                Number of test samples.
        """
        # return metrics, num_samples
        raise NotImplementedError("[Trainer::test]: Please Implement test() method")

    def gather_into_tensor(self, tensor: torch.Tensor):
        return self.training_strategy.gather_into_tensor(tensor)

    @property
    def actual_model(self):
        return self.training_strategy.get_actual_model(self)

    @property
    def actual_optimizer(self):
        return self.training_strategy.get_actual_optimizer(self)

    def log_to_wandb(self, *args, **kwargs):
        self.training_strategy.log_wandb(*args, **kwargs)

    def barrier(self, name=""):
        logging.info(f"[Trainer::barrier]: Reached barrier {name}. Waiting...")
        self.training_strategy.barrier()
        logging.info(f"[Trainer::barrier]: Barrier {name} passed.")
