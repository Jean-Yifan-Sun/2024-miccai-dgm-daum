import logging
import os
#
import matplotlib
import pandas as pd
import numpy as np
import subprocess

from dl_utils.image_utils import img_3d_to_2d, make_grid
from model_zoo.cardiac_classifier import CardiacClassifier
from optim.metrics.fid import FrechetInceptionDistance

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import nibabel as nib
import wandb
import math
import torch
from core.DownstreamEvaluator import DownstreamEvaluator
from core.Main import is_master
from pytorch_msssim import ms_ssim
from torchvision import transforms
from model_zoo.cardiac_classifier import CardiacClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from optim.metrics.lpips import LPIPS
#from frd_score import frd



class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, evaluation_mode, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, evaluation_mode, model, device, test_data_dict, checkpoint_path)
        self.fid_real = FrechetInceptionDistance()

        self.fid_fake1 = FrechetInceptionDistance()
        self.fid_fake2 = FrechetInceptionDistance()

        self.cardiac_classifier = CardiacClassifier(input_size=(1, 13, 96, 96),
                                                    input_dim=3,
                                                    layer_channels=8,
                                                    layer_multiplier=[1, 2, 3, 4],
                                                    layer_attention=[False] * 4,
                                                    residual_blocks=2,
                                                    sample_resnet=False,
                                                    dropout=0.1,
                                                    normalization='group',
                                                    normalization_groups=8)

        classifier_path = './projects/cardiac_classifier/weights/cardiac_classifier/2024_10_31_17_44_34_267009/latest_model.pt'
        checkpoint = torch.load(classifier_path, map_location=torch.device(self.device))
        self.cardiac_classifier.load_state_dict(checkpoint['model_weights'])
        self.cardiac_classifier.to(self.device)
        self.cardiac_classifier.eval()

        self.lpips_metric = LPIPS()


    def start_task(self, global_model):
        """
        :param global_model: dict
                   the model weights
        """

        self.actual_model.load_state_dict(global_model)
        self.model.eval()

        with torch.no_grad():
            ms_ssim_real = []
            ms_ssim_generated = []

            actual_labels = []
            predicted_labels = []

            dataset = self.test_data_dict['test']
            ### CALCULATE METRICS ###
            for i, data in enumerate(dataset):
                images, labels = data
                # logging.info(f'labels: {labels}')
                labels = labels.to(self.device)
                images = images.to(self.device)

                generated_images = self.actual_model.sample(len(labels), labels, device=self.device, latents=False).detach()
                # Set slices that have little variation to mean of other slices as these are most likely just padding
                std_dev = generated_images.std(dim=[2, 3], keepdim=True)
                empty_mask = std_dev <= 0.1

                generated_images = generated_images * (~empty_mask)
                generated_images = generated_images + (generated_images.mean() * empty_mask)

                # Gather so that main node can calculate metrics over all generations
                generated_images = self.gather_into_tensor(generated_images)

                labels = self.gather_into_tensor(labels)
                images = self.gather_into_tensor(images).cpu()

                if not is_master():
                    continue

                # Saves images so they can be segmented later
                self.save_batch_to_nifti(generated_images, labels)
                # Saves images for FRD calculation
                self.save_batch_for_frd(images[:, 3:10, ...], generated_images[:, 3:10, ...])

                labels_pred = self.cardiac_classifier(generated_images)
                labels_pred = (labels_pred > 0).int()

                ESED_index = self.actual_model.get_label_index('ESED')
                actual = labels[:, ESED_index].cpu().numpy()
                predicted = labels_pred.cpu().numpy()

                # Accumulate the labels for evaluation
                actual_labels.extend(actual)
                predicted_labels.extend(predicted)

                logging.info(f'labels: {labels.shape}, ESED_index: {ESED_index}, labels_pred: {labels_pred}')

                if i == 0:
                    self.evaluate_noising_denoising(images, labels)

                    image_grid = make_grid(generated_images.cpu().numpy())
                    wandb.log({'Test/Generated_': [
                        wandb.Image(image_grid, caption="Generated_")]})

                    es_mask = labels[:, ESED_index] == 0
                    generated_images_es = generated_images[es_mask]

                    # Filter images for ED
                    ed_mask = labels[:, ESED_index] == 1

                    logging.info(f'labels: {labels.shape}, es_mask: {es_mask.shape}, ed_mask: {ed_mask.shape}')
                    generated_images_ed = generated_images[ed_mask]

                    if len(generated_images_es) > 0:
                        image_grid_es = make_grid(generated_images_es.cpu().numpy())
                        wandb.log({'Test/Generated_ES': [
                            wandb.Image(image_grid_es, caption="Generated_ES")]})

                    # Create a grid for ED images and log them
                    if len(generated_images_ed) > 0:
                        image_grid_ed = make_grid(generated_images_ed.cpu().numpy())
                        wandb.log({'Test/Generated_ED': [
                            wandb.Image(image_grid_ed, caption="Generated_ED")]})

                fid_images = images[:, 3:10, ...]
                fid_images_1, fid_images_2 = fid_images.split(len(fid_images) // 2, dim=0)
                self.fid_real(fid_images_1, fid_images_2)

                fid_generated_images = generated_images[:, 3:10, ...]
                fid_generated_images_1, fid_generated_images_2 = fid_generated_images.split(len(fid_generated_images) // 2, dim=0)
                self.fid_fake1(fid_images_1, fid_generated_images_1)
                self.fid_fake2(fid_images_2, fid_generated_images_2)

                images = resize_for_ms_ssim(fid_images)                           # Changes dimension from 96x96 to 192x192
                generated_images = resize_for_ms_ssim(fid_generated_images.cpu())       #

                ms_ssim_real.append(ms_ssim(images, torch.roll(images, shifts=1, dims=0), data_range=1))
                ms_ssim_generated.append(
                    ms_ssim(generated_images, torch.roll(generated_images, shifts=1, dims=0), data_range=1))

            if not is_master():
                return

            # create table for FIDs
            df = pd.DataFrame()

            df['FID Real'] = [self.fid_real.calculate()]
            df['FID Generated'] = [np.mean([self.fid_fake1.calculate(), self.fid_fake2.calculate()])]
            df['MS-SSIM Real'] = [np.mean(ms_ssim_real)]
            df['MS-SSIM Generated'] = [np.mean(ms_ssim_generated)]

            actual_labels = np.array(actual_labels)
            predicted_labels = np.array(predicted_labels)

            # Calculate metrics
            cycle_accuracy = accuracy_score(actual_labels, predicted_labels)
            df['Cycle Accuracy'] = [cycle_accuracy]

            self.call_segmentation_script()
            self.fix_padding_segmentations()
            # calculate correlation between Volumes and Thickness
            df['LV Vol Correlation'], df['RV Vol Correlation'] = self.calculate_LV_RV_volume_correlation()

            self.call_thickness_script()
            df['Myo Thickness Correlation'] = self.calculate_wall_thickness_correlation()

            tbl = wandb.Table(data=df)
            wandb.log({'Test/Metrics': tbl})

    def evaluate_noising_denoising(self, images, labels):
        labels = labels.to(self.device)

        # Show noised and denoised images for all steps
        steps = torch.linspace(0, self.actual_model.T - 2, steps=self.actual_model.T - 1, device=self.device)
        steps = steps.round().int()

        image = images[0, ...].unsqueeze(0)
        if len(image.shape) == 4:
            repeated_images = image.repeat(len(steps), 1, 1, 1)
        elif len(image.shape) == 5:
            repeated_images = image.repeat(len(steps), 1, 1, 1, 1)
        else:
            raise ValueError('Image dimension must be 2D or 3D')

        x_t, epsilon = self.actual_model.diffusion_forward(repeated_images.to(self.device), steps.to(self.device))
        x_t = x_t.squeeze(1).cpu()

        stats = []

        with torch.no_grad():
            reverse_steps = self.actual_model.sample(len(labels), labels, device=self.device, diffusion_steps=True)
            reverse_steps = [rs.cpu() for rs in reverse_steps]

            for i, step in enumerate(steps):
                # Extract the image at the current step
                image = x_t[i]
                image_std = image.std()
                image_mean = image.mean()
                image_min = image.min()
                image_max = image.max()
                image = img_3d_to_2d(image)

                rev_image = reverse_steps[i]
                rev_image_std = rev_image.std()
                rev_image_mean = rev_image.mean()
                rev_image_min = rev_image.min()
                rev_image_max = rev_image.max()
                rev_image = img_3d_to_2d(rev_image[0, ...].squeeze(0))

                stats.append(
                    [i, image_mean, image_std, image_min, image_max, rev_image_mean, rev_image_std, rev_image_min,
                     rev_image_max])

                # stack on top of each other
                image = np.vstack((image, rev_image))
                # Log the image with wandb
                wandb.log({f'Test/Forward_Reverse': wandb.Image(image), 'global_step': step.item()})

        df = pd.DataFrame(stats,
                          columns=['Step', 'Mean', 'Std', 'Min', 'Max', 'Rev_Mean', 'Rev_Std', 'Rev_Min', 'Rev_Max'])
        tbl = wandb.Table(data=df)

        wandb.log({'Test/ProcessStats': tbl})

    def save_batch_for_frd(self, real_images, fake_images):
        real_images = real_images.squeeze().cpu().numpy()
        fake_images = fake_images.squeeze().cpu().numpy()

        frd_path = os.path.join(self.checkpoint_path, 'frd_images')
        real_images_path = os.path.join(frd_path, 'real_images')
        fake_images_path = os.path.join(frd_path, 'fake_images')

        if not os.path.exists(frd_path):
            os.makedirs(frd_path)
            os.makedirs(real_images_path)
            os.makedirs(fake_images_path)

        hash_keys = real_images[:, 0, 0, 0].tolist()
        for i in range(real_images.shape[0]):
            real_image = real_images[i]
            fake_image = fake_images[i]

            real_image = nib.Nifti1Image(real_image, affine=np.eye(4))
            fake_image = nib.Nifti1Image(fake_image, affine=np.eye(4))

            # Save the NIfTI image to disk
            image_name = str(abs(hash(tuple(hash_keys + [i]))))
            nib.save(real_image, os.path.join(real_images_path, f'{image_name}.nii.gz'))
            nib.save(fake_image, os.path.join(fake_images_path, f'{image_name}.nii.gz'))

    def calculate_frd_values(self):
        # Calculate FRD
        frd_path = os.path.join(self.checkpoint_path, 'frd_images')

        real_images_path = os.path.join(frd_path, 'real_images')
        fake_images_path = os.path.join(frd_path, 'fake_images')

        real_images = os.listdir(real_images_path)
        real_images = [os.path.join(real_images_path, f) for f in real_images]
        real_images_1, real_images_2 = real_images[:len(real_images) // 2], real_images[len(real_images) // 2:]

        fake_images = os.listdir(fake_images_path)
        fake_images = [os.path.join(fake_images_path, f) for f in fake_images]
        fake_images_1, fake_images_2 = fake_images[:len(fake_images) // 2], fake_images[len(fake_images) // 2:]

        frd_real = frd.compute_frd([real_images_1, real_images_2])

        frd_fake_1 = frd.compute_frd([real_images_1, fake_images_1])
        frd_fake_2 = frd.compute_frd([real_images_2, fake_images_2])
        frd_fake = (frd_fake_1 + frd_fake_2) / 2
        return frd_real, frd_fake

    def save_batch_to_nifti(self, generated_batch, labels):
        generated_batch = generated_batch.squeeze().cpu().numpy()
        labels = labels.cpu().numpy()
        # Transpose from B,C,H,W to B,H,W,C for segmentation network
        generated_batch = generated_batch.transpose(0, 2, 3, 1)

        hash_keys = generated_batch[:, 0, 0, 0].tolist()
        for i in range(generated_batch.shape[0] // 2):
            folder_name = str(abs(hash(tuple(hash_keys + [i]))))
            dir = os.path.join(self.image_path, folder_name)
            if not os.path.exists(dir):
                os.makedirs(dir)

            print("Saving to: ", dir)
            es_image = generated_batch[i * 2, ...]
            ed_image = generated_batch[i * 2 + 1, ...]

            es_labels = labels[i * 2, ...]
            ed_labels = labels[i * 2 + 1, ...]

            es_image = nib.Nifti1Image(es_image, affine=np.eye(4))
            ed_image = nib.Nifti1Image(ed_image, affine=np.eye(4))

            # Save the NIfTI image to disk
            nib.save(es_image, os.path.join(dir, 'sa_ES.nii.gz'))
            nib.save(ed_image, os.path.join(dir, 'sa_ED.nii.gz'))

            # Save labels
            np.save(os.path.join(dir, 'sa_ES_labels.npy'), es_labels)
            np.save(os.path.join(dir, 'sa_ED_labels.npy'), ed_labels)

    def call_segmentation_script(self):
        path_to_conda_activate = "<path>/miniconda3/bin/activate"
        activate_env = f"source {path_to_conda_activate} pySegment2"
        change_dir = "cd <projects_dir>"
        run_script = f'PYTHONPATH="$PWD:$PYTHONPATH" python ukbb_cardiac/common/deploy_network.py --seq_name sa --noprocess_seq --model_path <model_path> --data_dir {os.path.abspath(self.image_path)}'
        command = f'{activate_env} && {change_dir} && {run_script}'

        logging.info("Launching segmentation script")
        process = subprocess.run(command, shell=True, executable='/bin/bash', text=True, capture_output=True)
        logging.info("Segmentation process completed")

        if process.returncode == 0:
            logging.info("Segmentation completed successfully.")
            return True
        else:
            logging.info("Error in Segmentation.")
            logging.info(process.stderr)
            return False

    def fix_padding_segmentations(self):
        logging.info("Fixing padding segmentations")
        folders = os.listdir(self.image_path)
        for f in folders:
            folder_path = os.path.join(self.image_path, f)
            for c in ['ES', 'ED']:
                image_path = os.path.join(folder_path, f'sa_{c}.nii.gz')
                segmentation_path = os.path.join(folder_path, f'seg_sa_{c}.nii.gz')
                if not os.path.exists(image_path) or not os.path.exists(segmentation_path):
                    continue
                segmentation = nib.load(segmentation_path)
                segmentation_data = segmentation.get_fdata()

                image_data = nib.load(image_path).get_fdata()

                segmentation_data[..., image_data.std(axis=(0, 1)) < 0.1] = 0
                new_segmentation = nib.Nifti1Image(segmentation_data, affine=segmentation.affine)
                nib.save(new_segmentation, segmentation_path)

        logging.info("Fixing padding segmentations done!")

    def call_thickness_script(self):
        path_to_conda_activate = "<path>/miniconda3/bin/activate"
        activate_env = f"source {path_to_conda_activate} pySegment2"
        change_dir = "cd <projects_dir>"
        run_script = f'PYTHONPATH="$PWD:$PYTHONPATH" python myo_evaluation.py --path {os.path.abspath(self.image_path)}'
        command = f'{activate_env} && {change_dir} && {run_script}'

        logging.info("Launching wall thickness script")
        process = subprocess.run(command, shell=True, executable='/bin/bash', text=True, capture_output=True)
        logging.info("Wall thickness process completed")

        if process.returncode == 0:
            logging.info("Wall thickness completed successfully.")
            logging.info(process.stderr)
            logging.info(process.stdout)
            return True
        else:
            logging.info("Error in wall thickness.")
            logging.info(process.stderr)
            return False

    def calculate_wall_thickness_correlation(self):
        folders = os.listdir(self.image_path)
        thicknesses = []
        for f in folders:
            folder_path = os.path.join(self.image_path, f)
            for c in ['ES', 'ED']:
                label_path = os.path.join(folder_path, f'sa_{c}_labels.npy')
                thickness_path = os.path.join(folder_path, f'MYO{c}THICK.npy')
                if not os.path.exists(thickness_path) or not os.path.exists(label_path):
                    continue
                thickness = np.load(thickness_path)
                label = np.load(label_path)
                ESED = label[self.actual_model.get_label_index('ESED')]
                label_thickness = label[self.actual_model.get_label_index('MYOESTHICK')] if ESED == 0 else label[self.actual_model.get_label_index('MYOEDTHICK')]
                thicknesses.append([thickness, label_thickness])

        corr = np.corrcoef(np.array(thicknesses).T)[0, 1]
        return corr

    def calculate_LV_RV_volume_correlation(self):
        folders = os.listdir(self.image_path)
        pixel_volume = 1.8269 * 1.8269 * 10.0
        lv_values = []
        rv_values = []
        for f in folders:
            folder_path = os.path.join(self.image_path, f)
            for c in ['ES', 'ED']:
                seg = nib.load(os.path.join(folder_path, f'seg_sa_{c}.nii.gz')).get_fdata().transpose(2, 0, 1)
                lab = np.load(os.path.join(folder_path, f'sa_{c}_labels.npy'))
                lv = (seg == 1).sum() * pixel_volume
                rv = (seg == 3).sum() * pixel_volume
                l_lv = lab[self.actual_model.get_label_index('LVESV')] if lab[self.actual_model.get_label_index('ESED')] == 0 else lab[self.actual_model.get_label_index('LVEDV')]
                l_rv = lab[self.actual_model.get_label_index('RVESV')] if lab[self.actual_model.get_label_index('ESED')] == 0 else lab[self.actual_model.get_label_index('RVEDV')]
                lv_values.append([lv, l_lv])
                rv_values.append([rv, l_rv])
        lv_corr = np.corrcoef(np.array(lv_values).T)[0, 1]
        rv_corr = np.corrcoef(np.array(rv_values).T)[0, 1]
        return lv_corr, rv_corr


def resize_for_ms_ssim(image):
    if len(image.shape) != 4:
        raise ValueError('Only defined for 4 Channels')

    B, C, H, W = image.shape
    if H > 160:
        return image

    resize_size = int((2**math.ceil(math.log2(160 / H))) * H)
    resize = transforms.Resize(resize_size)

    return resize(image)






