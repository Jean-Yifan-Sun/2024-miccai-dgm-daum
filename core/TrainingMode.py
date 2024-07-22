from abc import ABC
from enum import Enum
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb


class TrainingMode(Enum):
    STANDARD = 0,
    DATA_PARALLEL = 1
    DISTRIBUTED_DATA_PARALLEL = 2
    STANDARD_DP = 3
    DISTRIBUTED_DP = 4

    def get_strategy(self):
        if self == TrainingMode.STANDARD:
            return StandardTrainingStrategy()
        elif self == TrainingMode.DATA_PARALLEL:
            return DataParallelTrainingStrategy()
        elif self == TrainingMode.DISTRIBUTED_DATA_PARALLEL:
            return DDPTrainingStrategy()
        elif self == TrainingMode.STANDARD_DP:
            return StandardDPTrainingMode()
        elif self == TrainingMode.DISTRIBUTED_DP:
            return DistributedDPTrainingMode()
        else:
            raise AttributeError("Unknown training mode.")

    def is_DP(self):
        return self == TrainingMode.STANDARD_DP or self == TrainingMode.DISTRIBUTED_DP


class AbstractTrainingStrategy(ABC):
    def __init__(self):
        pass

    def prepare_trainer(self, trainer):
        raise NotImplementedError

    def gather_into_tensor(self, tensor: torch.Tensor):
        raise NotImplementedError

    def get_actual_model(self, trainer):
        raise NotImplementedError

    def get_actual_optimizer(self, trainer):
        raise NotImplementedError

    def log_wandb(self, *args, **kwargs):
        raise NotImplementedError

    def barrier(self):
        raise NotImplementedError


class StandardTrainingStrategy(AbstractTrainingStrategy):
    def __init__(self):
        super(StandardTrainingStrategy, self).__init__()

    def prepare_trainer(self, trainer):
        pass

    def gather_into_tensor(self, tensor: torch.Tensor):
        return tensor

    def get_actual_model(self, trainer):
        return trainer.model

    def get_actual_optimizer(self, trainer):
        return trainer.optimizer

    def log_wandb(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def barrier(self):
        pass


class DataParallelTrainingStrategy(AbstractTrainingStrategy):
    def __init__(self):
        super(DataParallelTrainingStrategy, self).__init__()

    def prepare_trainer(self, trainer):
        logging.info("[Configurator::init:INFO]: wrapping model for Data Parallel")
        trainer.model = torch.nn.DataParallel(trainer.model)

    def gather_into_tensor(self, tensor: torch.Tensor):
        return tensor

    def get_actual_model(self, trainer):
        return trainer.model.module

    def get_actual_optimizer(self, trainer):
        return trainer.optimizer

    def log_wandb(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def barrier(self):
        pass


class DDPTrainingStrategy(AbstractTrainingStrategy):
    def __init__(self):
        super(DDPTrainingStrategy, self).__init__()

    def prepare_trainer(self, trainer):
        logging.info("[Configurator::init:INFO]: wrapping model")
        trainer.model = DDP(trainer.model, device_ids=[0], find_unused_parameters=True)
        logging.info("[Configurator::init:INFO]: wrapping dataloaders")
        trainer.train_ds = _make_distributed(trainer.train_ds)
        trainer.val_ds = _make_distributed(trainer.val_ds)
        trainer.test_ds = _make_distributed(trainer.test_ds)
        logging.info("[Configurator::init:INFO]: Done!")

    def gather_into_tensor(self, tensor: torch.Tensor):
        """
            :param tensor: Tensor to gather from all processes
        """
        tensor_shape = list(tensor.shape)
        tensor_shape[0] *= dist.get_world_size()

        all_tensor = torch.zeros(*tensor_shape, device=tensor.device, dtype=tensor.dtype)
        dist.all_gather_into_tensor(all_tensor, tensor)
        return all_tensor

    def get_actual_model(self, trainer):
        return trainer.model.module

    def get_actual_optimizer(self, trainer):
        return trainer.optimizer

    def log_wandb(self, *args, **kwargs):
        if dist.get_rank() == 0:
            wandb.log(*args, **kwargs)

    def barrier(self):
        torch.distributed.barrier()

class StandardDPTrainingMode(StandardTrainingStrategy):
    def __init__(self):
        super(StandardDPTrainingMode, self).__init__()


    def prepare_trainer(self, trainer):
        from opacus import PrivacyEngine
        logging.info("[Configurator::init:INFO]: wrapping model for differential privacy")
        trainer.privacy_engine = PrivacyEngine()
        trainer.model.train()
        trainer.model, trainer.optimizer, trainer.train_ds = trainer.privacy_engine.make_private_with_epsilon(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.train_ds,
            #noise_multiplier=1.,
            epochs=trainer.training_params['nr_epochs'],
            target_delta=trainer.training_params['differential_privacy']['target_delta'],
            target_epsilon=trainer.training_params['differential_privacy']['target_epsilon'],
            max_grad_norm=trainer.training_params['differential_privacy']['max_grad_norm']
        )

        logging.info(
            f"After privatization: model ({type(trainer.model).__name__}), "
            f"optimizer ({type(trainer.optimizer).__name__}), "
            f"data loader ({type(trainer.train_ds).__name__}, len={len(trainer.train_ds)})"
        )

    def get_actual_model(self, trainer):
        return trainer.model._module

    def get_actual_optimizer(self, trainer):
        return trainer.optimizer.original_optimizer


class DistributedDPTrainingMode(DDPTrainingStrategy):
    def __init__(self):
        super(DistributedDPTrainingMode, self).__init__()

    def prepare_trainer(self, trainer):
        from opacus import PrivacyEngine
        from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
        logging.info("[Configurator::init:INFO]: wrapping model for DPDDP")
        trainer.privacy_engine = PrivacyEngine()
        trainer.model.train()
        trainer.model = DPDDP(trainer.model)
        trainer.model, trainer.optimizer, trainer.train_ds = trainer.privacy_engine.make_private_with_epsilon(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.train_ds,
            #noise_multiplier=1.,
            epochs=trainer.training_params['nr_epochs'],
            target_delta=trainer.training_params['differential_privacy']['target_delta'],
            target_epsilon=trainer.training_params['differential_privacy']['target_epsilon'],
            max_grad_norm=trainer.training_params['differential_privacy']['max_grad_norm'],
        )
        self.barrier()
        trainer.val_ds = _make_distributed(trainer.val_ds)
        trainer.test_ds = _make_distributed(trainer.test_ds)

        logging.info(
            f"(rank {dist.get_rank()}) After privatization: model ({type(trainer.model).__name__}), "
            f"optimizer ({type(trainer.optimizer).__name__}), "
            f"data loader ({type(trainer.train_ds).__name__}, len={len(trainer.train_ds)})"
        )

        logging.info(f"(rank {dist.get_rank()}) Average batch size per GPU: {int(trainer.optimizer.expected_batch_size)}")

    def get_actual_optimizer(self, trainer):
        return trainer.optimizer.original_optimizer



def _make_distributed(dataloader):
    assert dataloader.batch_size % dist.get_world_size() == 0, "Batch size must be divisible by the number of nodes"
    logging.info(f"batch size per GPU: {dataloader.batch_size // dist.get_world_size()}")

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataloader.dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank()
    )
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size // dist.get_world_size(),
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        sampler=sampler
    )