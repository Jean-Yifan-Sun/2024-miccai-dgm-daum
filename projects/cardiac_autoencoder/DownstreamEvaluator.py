import logging
#
import matplotlib
import pandas as pd

from dl_utils.image_utils import img_3d_to_2d, make_grid
from model_zoo.autoencoder import Action
from optim.metrics.fid import FrechetInceptionDistance

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
from dl_utils import *
from core.DownstreamEvaluator import DownstreamEvaluator
from optim.metrics.lpips import LPIPS

from skimage.metrics import structural_similarity
from torch.nn import MSELoss


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, evaluation_mode, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, evaluation_mode, model, device, test_data_dict, checkpoint_path)

        self.criterion_mse = MSELoss()
        self.lpips_metric = LPIPS()

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.actual_model.load_state_dict(global_model)
        self.model.eval()
        dataset = self.test_data_dict['test']

        metrics = {'MSE': [],
                   'LPIPS': [],
                   'SSIM': [],
                   'Latent_Std': [],
                   'Latent_Mean': []}

        with torch.no_grad():
            for idx, data in enumerate(dataset):
                # Input
                batch = [b.to(self.device) for b in data]
                # process batch data
                images, labels = batch

                x_recon, dist = self.model(images)
                z = self.model(images, Action.ENCODE).squeeze(1)

                ### Latent Mean ###
                metrics['Latent_Std'].append(z.std().cpu().item())

                ### Latent Mean ###
                metrics['Latent_Mean'].append(z.mean().cpu().item())

                ### MSE ###
                metrics['MSE'].append(self.criterion_mse(images, x_recon).cpu().item())

                ### LPIPS ###
                metrics['LPIPS'].append(self.lpips_metric(images, x_recon).mean().cpu().item())

                ### SSIM ###
                images = images.cpu().numpy()
                x_recon = x_recon.cpu().numpy()
                ssim_list = []
                for i in range(len(images)):
                    ssim_list.append(structural_similarity(images[i], x_recon[i], data_range=1.0))

                metrics['SSIM'].append(np.mean(ssim_list))

        for k in metrics.keys():
            metrics[k] = np.mean(metrics[k])

        df_metrics = pd.DataFrame(list([metrics.values()]), columns=list(metrics.keys()))
        tbl = wandb.Table(data=df_metrics)
        wandb.log({f"Test/Metrics": tbl})

