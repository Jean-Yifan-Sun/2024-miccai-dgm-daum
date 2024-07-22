from abc import ABC
from enum import Enum
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb


class EvaluationMode(Enum):
    STANDARD = 0,
    DATA_PARALLEL = 1
    DISTRIBUTED_DATA_PARALLEL = 2

    def get_strategy(self):
        if self == EvaluationMode.STANDARD:
            return StandardEvaluationStrategy()
        elif self == EvaluationMode.DATA_PARALLEL:
            return DataParallelEvaluationStrategy()
        elif self == EvaluationMode.DISTRIBUTED_DATA_PARALLEL:
            return DDPEvaluationStrategy()
        else:
            raise AttributeError("Unknown training mode.")

class AbstractEvaluationStrategy(ABC):
    def __init__(self):
        pass

    def prepare_evaluator(self, evaluator):
        raise NotImplementedError

    def gather_into_tensor(self, tensor: torch.Tensor):
        raise NotImplementedError

    def get_actual_model(self, evaluator):
        raise NotImplementedError

    def log_wandb(self, *args, **kwargs):
        raise NotImplementedError

    def barrier(self):
        raise NotImplementedError


class StandardEvaluationStrategy(AbstractEvaluationStrategy):
    def __init__(self):
        super(StandardEvaluationStrategy, self).__init__()

    def prepare_evaluator(self, evaluator):
        pass

    def gather_into_tensor(self, tensor: torch.Tensor):
        return tensor

    def get_actual_model(self, evaluator):
        return evaluator.model

    def log_wandb(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def barrier(self):
        pass


class DataParallelEvaluationStrategy(AbstractEvaluationStrategy):
    def __init__(self):
        super(DataParallelEvaluationStrategy, self).__init__()

    def prepare_evaluator(self, evaluator):
        logging.info("[Configurator::init:INFO]: wrapping model for DataParallel")
        evaluator.model = torch.nn.DataParallel(evaluator.model)

    def gather_into_tensor(self, tensor: torch.Tensor):
        return tensor

    def get_actual_model(self, evaluator):
        return evaluator.model.module

    def log_wandb(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def barrier(self):
        pass


class DDPEvaluationStrategy(AbstractEvaluationStrategy):
    def __init__(self):
        super(DDPEvaluationStrategy, self).__init__()

    def prepare_evaluator(self, evaluator):
        logging.info("[Configurator::init:INFO]: wrapping dataloaders")
        for name in evaluator.test_data_dict:
            evaluator.test_data_dict[name] = _make_distributed(evaluator.test_data_dict[name])
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

    def get_actual_model(self, evaluator):
        return evaluator.model

    def log_wandb(self, *args, **kwargs):
        if dist.get_rank() == 0:
            wandb.log(*args, **kwargs)

    def barrier(self):
        torch.distributed.barrier()


def _make_distributed(dataloader,):
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