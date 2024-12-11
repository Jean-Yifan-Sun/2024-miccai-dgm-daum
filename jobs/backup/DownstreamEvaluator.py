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


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, evaluation_mode, model, device, test_data_dict, checkpoint_path, **config):
        super(PDownstreamEvaluator, self).__init__(name, evaluation_mode, model, device, test_data_dict, checkpoint_path)

        self.esed_dim = 4


    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.model.module.load_state_dict(global_model)
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
                labels = labels[:, self.esed_dim]

                # Forward pass
                labels_pred = self.model(images)
                labels_pred = (labels_pred > 0).int()

                all_preds.append(labels_pred.cpu())
                all_labels.append(labels.cpu())

                test_total += images.size(0)

        images_es = images[labels_pred == 0].detach().cpu().numpy()
        images_ed = images[labels_pred == 1].detach().cpu().numpy()

        if len(images_es) > 0:
            image_grid = make_grid(images_es)
            wandb.log({'Test/Predicted_ES': [
                wandb.Image(image_grid, caption="Iteration_")]})

        if len(images_ed) > 0:
            image_grid = make_grid(images_ed)
            wandb.log({'Test/Predicted_ED': [
                wandb.Image(image_grid, caption="Iteration_")]})

        ### Calculate Metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        accuracy = (all_preds == all_labels).float().mean().item()
        cm = confusion_matrix(all_labels, all_preds)

        fig = plot_confusion_matrix(cm, class_names=['ES', 'ED'])

        metrics_data = {
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall]
        }

        df_metrics = pd.DataFrame(metrics_data)
        tbl = wandb.Table(data=df_metrics)

        wandb.log({'Test/Metrics': tbl, 'Test/Confusion Matrix': wandb.Image(fig)})


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

