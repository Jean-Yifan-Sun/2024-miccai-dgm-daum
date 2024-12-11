import logging
#
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from dl_utils.image_utils import make_grid

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
from dl_utils import *
from core.DownstreamEvaluator import DownstreamEvaluator


class PDownstreamEvaluator3D(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, evaluation_mode, model, device, test_data_dict, checkpoint_path, **config):
        super(PDownstreamEvaluator3D, self).__init__(name, evaluation_mode, model, device, test_data_dict, checkpoint_path)

        self.esed_dim = 4

    def toSlices(self, x):
        # logging.info(f'x.shape: {x.shape}')
        assert len(x.shape) == 4
        # take slices
        slices = []
        for i in range(13):
            slices.append(x[:,i,:,:])
        slices = torch.cat(slices,dim=0)
        return slices

    def unSlice(self,x):
        assert len(x.shape) == 3
        num_slices = x.shape[0]//13
        unpack_batch = []
        for i in range(num_slices):
            
            unpack = []
            for j in range(13):
                slice = x[i*13+j,:,:]
                frame = slice.unsqueeze(0)
                unpack.append(frame)
            unpack = torch.cat(unpack,dim=0)
            unpack_batch.append(unpack.unsqueeze(0))
        unpack_batch = torch.cat(unpack_batch,dim=0)
        return unpack_batch

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.model.load_state_dict(global_model)
        self.model.eval()
        dataset = self.test_data_dict['test']

        all_preds = []
        all_labels = []

        test_total = 0
        with torch.no_grad():
            for idx, data in enumerate(dataset):
                # Input
                batch = [b.to(self.device) for b in data]
                # process batch data
                images, labels = batch
                # labels = labels[:, self.esed_dim]

                # Forward pass
                labels_pred = self.model(images)
                
                labels_pred = (labels_pred > 0).int()

                all_preds.append(labels_pred.cpu())
                all_labels.append(labels.cpu().int())

                test_total += images.size(0)

        images_es = images*(labels_pred == 0)
        images_es = images_es.detach().cpu().numpy()
        images_ed = images*(labels_pred == 1)
        images_ed = images_ed.detach().cpu().numpy()

        if len(images_es) > 0:
            image_grid = make_grid(images_es)
            wandb.log({'Test/Predicted_ES': [
                wandb.Image(image_grid, caption="Iteration_")]})

        if len(images_ed) > 0:
            image_grid = make_grid(images_ed)
            wandb.log({'Test/Predicted_ED': [
                wandb.Image(image_grid, caption="Iteration_")]})

        ### Calculate Metrics
        # all_preds = torch.cat(all_preds)
        # all_labels = torch.cat(all_labels)

        # precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        # accuracy = (all_preds == all_labels).float().mean().item()
        # cm = confusion_matrix(all_labels, all_preds)

        # fig = plot_confusion_matrix(cm, class_names=['ES', 'ED'])

        # metrics_data = {
        #     'Accuracy': [accuracy],
        #     'Precision': [precision],
        #     'Recall': [recall]
        # }

        # df_metrics = pd.DataFrame(metrics_data)
        # tbl = wandb.Table(data=df_metrics)

        # wandb.log({'Test/Metrics': tbl, 'Test/Confusion Matrix': wandb.Image(fig)})
        for i in range(len(all_labels)):
            miou = calculate_miou(all_preds[i],all_labels[i],num_classes=2)
            dice = calculate_dice(all_preds[i],all_labels[i])
            logging.info(f'miou: {miou}, dice: {dice}')
        save_imgs(all_labels,'/rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/projects/cardiac_classifier/output3d/labels')
        save_imgs(all_preds,'/rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/projects/cardiac_classifier/output3d/preds')


class PDownstreamEvaluator2D(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, evaluation_mode, model, device, test_data_dict, checkpoint_path, **config):
        super(PDownstreamEvaluator2D, self).__init__(name, evaluation_mode, model, device, test_data_dict, checkpoint_path)

        self.esed_dim = 4

    def toSlices(self, x):
        # logging.info(f'x.shape: {x.shape}')
        assert len(x.shape) == 4
        # take slices
        slices = []
        for i in range(13):
            slices.append(x[:,i,:,:])
        slices = torch.cat(slices,dim=0)
        return slices

    def unSlice(self,x):
        assert len(x.shape) == 3
        num_slices = x.shape[0]//13
        unpack_batch = []
        for i in range(num_slices):
            
            unpack = []
            for j in range(13):
                slice = x[i*13+j,:,:]
                frame = slice.unsqueeze(0)
                unpack.append(frame)
            unpack = torch.cat(unpack,dim=0)
            unpack_batch.append(unpack.unsqueeze(0))
        unpack_batch = torch.cat(unpack_batch,dim=0)
        return unpack_batch

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.model.load_state_dict(global_model)
        self.model.eval()
        dataset = self.test_data_dict['test']

        all_preds = []
        all_labels = []

        test_total = 0
        with torch.no_grad():
            for idx, data in enumerate(dataset):
                # Input
                batch = [b.to(self.device) for b in data]
                # process batch data
                images, labels = batch
                # labels = labels[:, self.esed_dim]

                # Forward pass
        
                images = self.toSlices(images)
                # labels = self.toSlices(labels)
                labels_pred = self.model(images)
                images = self.unSlice(images)
                labels_pred = self.unSlice(labels_pred)
               
                
                labels_pred = (labels_pred > 0).int()

                all_preds.append(labels_pred.cpu())
                all_labels.append(labels.cpu().int())

                test_total += images.size(0)

        images_es = images*(labels_pred == 0)
        images_es = images_es.detach().cpu().numpy()
        images_ed = images*(labels_pred == 1)
        images_ed = images_ed.detach().cpu().numpy()

        if len(images_es) > 0:
            image_grid = make_grid(images_es)
            wandb.log({'Test/Predicted_ES': [
                wandb.Image(image_grid, caption="Iteration_")]})

        if len(images_ed) > 0:
            image_grid = make_grid(images_ed)
            wandb.log({'Test/Predicted_ED': [
                wandb.Image(image_grid, caption="Iteration_")]})

        ### Calculate Metrics
        # all_preds = torch.cat(all_preds)
        # all_labels = torch.cat(all_labels)

        # precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        # accuracy = (all_preds == all_labels).float().mean().item()
        # cm = confusion_matrix(all_labels, all_preds)

        # fig = plot_confusion_matrix(cm, class_names=['ES', 'ED'])

        # metrics_data = {
        #     'Accuracy': [accuracy],
        #     'Precision': [precision],
        #     'Recall': [recall]
        # }

        # df_metrics = pd.DataFrame(metrics_data)
        # tbl = wandb.Table(data=df_metrics)

        # wandb.log({'Test/Metrics': tbl, 'Test/Confusion Matrix': wandb.Image(fig)})
        for i in range(len(all_labels)):
            miou = calculate_miou(all_preds[i],all_labels[i],num_classes=2)
            dice = calculate_dice(all_preds[i],all_labels[i])
            logging.info(f'miou: {miou}, dice: {dice}')
        save_imgs(all_labels,'/rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/projects/cardiac_classifier/output/labels')
        save_imgs(all_preds,'/rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/projects/cardiac_classifier/output/preds')



def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using matplotlib

    :param cm: Confusion matrix
    :param class_names: Names of the classes in the dataset
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

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