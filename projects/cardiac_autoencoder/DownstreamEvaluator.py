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
                   'Latent_Mean': [],
                   'DICE':[]}

        self.all_recons,self.all_imgs = [],[]
        with torch.no_grad():
            for idx, data in enumerate(dataset):
                # Input
                batch = [b.to(self.device) for b in data]
                # process batch data
                images, labels = batch
                self.all_imgs.append(images)
                x_recon, dist = self.model(images)
                z = self.model(images, Action.ENCODE).squeeze(1)
                self.all_recons.append(x_recon)
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

        for i in range(len(self.all_imgs)):
            metrics['DICE'].append(calculate_dice(self.all_recons[i],self.all_imgs[i]))
            
        for k in metrics.keys():
            metrics[k] = np.mean(metrics[k])

        df_metrics = pd.DataFrame(list([metrics.values()]), columns=list(metrics.keys()))
        tbl = wandb.Table(data=df_metrics)
        wandb.log({f"Test/Metrics": tbl})
        df_metrics.to_csv(f'./projects/cardiac_autoencoder/Test_Metrics_{self.name}.csv')
        save_imgs(self.all_imgs,f'./projects/cardiac_autoencoder/{self.name}Images')
        save_imgs(self.all_recons,f'./projects/cardiac_autoencoder/{self.name}Recons')


def calculate_miou(pred_mask, true_mask, num_classes):
    miou = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()

        intersection = torch.sum(pred_cls * true_cls)
        union = torch.sum(pred_cls + true_cls) - intersection

        if union == 0:
            iou = 1  # Perfect match for empty classes
        else:
            iou = intersection / union

        miou.append(iou.item())
    return sum(miou) / len(miou)

def calculate_dice(prediction, ground_truth):
    # Flatten the tensors
    prediction = prediction.contiguous().view(-1)
    ground_truth = ground_truth.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(prediction * ground_truth)
    union = torch.sum(prediction) + torch.sum(ground_truth)
    
    # Avoid division by zero
    if union.item() == 0:
        return 1.0 if torch.sum(ground_truth).item() == 0 else 0.0
    
    # Calculate Dice score
    dice = (2.0 * intersection) / union
    return dice.item()

def save_imgs(imgs,path):
    import os
    import nibabel as nib
    import numpy as np
    os.makedirs(name=path,exist_ok=True)

    img_list = []
    if type(imgs) == list:
        num_batches = len(imgs)
        for i in range(num_batches):
            numpy_image = imgs[i].detach().cpu().numpy()
            if len(numpy_image.shape) == 4:
                for j in range(numpy_image.shape[0]):
                    img_list.append(numpy_image[j,:,:,:])
            elif len(numpy_image.shape) == 3:
                img_list.append(numpy_image)
    elif type(imgs) == torch.Tensor:
        numpy_image = imgs.detach().cpu().numpy()
        if len(numpy_image.shape) == 4:
            for j in range(numpy_image.shape[0]):
                img_list.append(numpy_image[j,:,:,:])
        elif len(numpy_image.shape) == 3:
            img_list.append(numpy_image)

    # Create an affine matrix (identity for simplicity)
    affine = np.eye(4)

    # Create a NIfTI image
    for i in range(len(img_list)):
        nii_image = nib.Nifti1Image(img_list[i], affine)
        # Save to a .nii.gz file
        nib.save(nii_image, os.path.join(path, f'output_image_{i}.nii.gz'))