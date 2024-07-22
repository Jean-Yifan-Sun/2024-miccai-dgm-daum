from random import random

import pandas as pd

from core.Trainer import Trainer
from core.Main import is_master, get_rank
from time import time
import wandb
import logging
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchinfo import summary
import torch.distributed as dist

from skimage.metrics import structural_similarity as ssim
import lpips

from dl_utils.image_utils import img_3d_to_2d, make_grid
from net_utils.diffusion.attention import Attention


class PTrainer(Trainer):
    def __init__(self, training_params, training_mode, model, data, device, log_wandb=True):
        self.finetune = 'finetune' in training_params and training_params['finetune']
        model.finetune(self.finetune)
        super(PTrainer, self).__init__(training_params, training_mode, model, data, device, log_wandb)
        self.lr_scheduler = StepLR(optimizer=self.optimizer, gamma=0.9, step_size=25) # 0.95
        self.context_disabled = 'context_disabled' in self.training_params and self.training_params['context_disabled']

        image_size = (2, 13, 96,  96)
        label_size = (2, len(self.actual_model.attribute_transformer.label_mapping) if self.actual_model.attribute_transformer is not None else 2)
        input_size = [image_size, label_size]
        logging.info(f"[Trainer::summary]: Model in: {input_size}")
        logging.info(f"[Trainer::summary]: {str(summary(self.actual_model, input_size))}")


    def train(self, model_state=None, opt_state=None, accountant=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.actual_model.load_state_dict(model_state)  # load weights
            logging.info("[Trainer::test]: Weights loaded")

        if 'start_epoch' in self.training_params:
            start_epoch = self.training_params['start_epoch']
        else:
            if self.finetune:
                logging.info("[Trainer::test]: Finetuning")
                start_epoch = 0
            if opt_state is not None:
                self.actual_optimizer.load_state_dict(opt_state)  # load optimizer
                logging.info("[Trainer::test]: Optimizers loaded")

        if accountant is not None:
            self.privacy_engine.accountant = accountant
        epoch_losses = []
        self.early_stop = False

        # disable training for image to latent model
        for param in self.actual_model.first_stage_model.parameters():
            param.requires_grad = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            logging.info("[Trainer::test]: Epoch {}".format(epoch))
            start_time = time()
            (batch_loss, count_images) = 0, 0
            for idx, data in enumerate(self.train_ds):
                # Input
                batch = [b.to(self.device) for b in data]
                # process batch data
                images, labels = batch
                if self.context_disabled:
                    labels = None
                mask = None

                if epoch == 0 and idx == 0:
                    latent_std = self.actual_model.calculate_latent_std(images).unsqueeze(-1)
                    latent_std = self.gather_into_tensor(latent_std).mean()
                    self.actual_model.set_latent_std(latent_std)
                    logging.info(f"[Trainer::train]: std set to {latent_std}")
                    if is_master():
                        self.show_noising_process(images[0, ...])

                ### Train###
                self.optimizer.zero_grad()

                if self.training_params['label_dropout']:
                    if random() > 0.9:
                        labels = None

                epsilon_pred, epsilon, t = self.model(images, labels, mask)

                # Reconstruction loss
                loss = self.criterion_rec(epsilon_pred, epsilon)
                loss.backward()

                self.optimizer.step()

                # Log
                batch_loss += loss.item() * images.size(0)
                count_images += images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            epoch_duration = end_time - start_time
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, epoch_duration, count_images))

            samples_per_second = count_images / epoch_duration  # Calculate samples per second
            # Log epoch duration and samples per second to wandb

            if self.training_mode.is_DP():
                DELTA = 1e-5
                epsilon = self.privacy_engine.get_epsilon(DELTA)
                logging.info(f"Differential privacy (ε = {epsilon:.2f}, δ = {DELTA})")

            if is_master():
                wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch, "Train/Epoch_Duration": epoch_duration,
                           "Train/Samples_per_Second": samples_per_second})

                # Save latest model
                save_object = {'model_weights': self.actual_model.state_dict(),
                                'optimizer_weights': self.actual_optimizer.state_dict(),
                                'epoch': epoch}
                if self.training_mode.is_DP():
                    save_object['accountant'] = self.privacy_engine.accountant

                torch.save(save_object, self.client_path + '/latest_model.pt')

            if epoch % 25 == 0 and epoch > 0:
                # Run validation
                self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.actual_model.state_dict(), self.actual_model.state_dict()

    def show_noising_process(self, image):
        steps = torch.linspace(0, self.actual_model.T - 1, steps=20, device=self.device)
        steps = steps.round().long()

        image = image.unsqueeze(0)
        if len(image.shape) == 4:
            repeated_images = image.repeat(len(steps), 1, 1, 1)
        else:
            repeated_images = image.repeat(len(steps), 1, 1, 1, 1)

        x_t, epsilon = self.actual_model.diffusion_forward(repeated_images, steps)
        x_t = x_t.squeeze(1)
        x_t = x_t.cpu()

        for i, step in enumerate(steps):
            # Extract the image at the current step
            image = x_t[i]
            image = img_3d_to_2d(image)
            # Log the image with wandb
            wandb.log({f'Forward/Images': wandb.Image(image), 'global_step': step.item()})

        final_mean = x_t[-1].mean()
        final_std = x_t[-1].std()

        df = pd.DataFrame([[final_mean, final_std]], columns=['Fwd Mean', 'Fwd Std'])
        tbl = wandb.Table(data=df)
        wandb.log({'Forward/Stats': tbl})


    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """

        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
        }

        # Handcrafted context to test rotation, ES/ED and slice conditioning for fixed attributes
        context = torch.tensor(
            [[142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 0.0, -45, 7.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 0.0, -45, 8.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 0.0, -25, 9.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 0.0, -15, 10.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 0.0, -5, 11.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 1.0, 5, 12.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 1.0, 15, 13.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 1.0, 25, 7.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 1.0, 35, 8.00],
             [142.97, 51.55, 63.93, 132.26, 48.83, 63.07, 10.38, 6.75, 66.0, 1.0, 27.69, 1.0, 45, 9.00],
             ]
        ).float().to(self.device)

        if self.training_params['label_dropout']:
            mask = torch.zeros_like(context, dtype=torch.bool)
        else:
            mask = None

        self.actual_model.eval()
        test_total = 0
        with torch.no_grad():
            # Generate new image
            images = self.actual_model.sample(len(context), context, mask=mask, device=self.device, latents=False)

            rec = images.detach().cpu().numpy()
            print("Min: {}, Max: {}, Mean: {}, Std: {}".format(rec.min(), rec.max(), rec.mean(), rec.std()))

            image_grid = make_grid(rec)
            self.log_to_wandb({task + '/Generated_': [
                wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})


            images_first, labels_first = None, None
            for data in test_data:
                batch = [b.to(self.device) for b in data]
                images, labels = batch
                if self.context_disabled:
                    labels = None
                if images_first is None:
                    images_first = images
                    labels_first = labels

                # Forward pass
                epsilon_pred, epsilon, t = self.model(images, labels)

                loss_rec = self.criterion_rec(epsilon_pred, epsilon)
                metrics[task + '_loss_rec'] += loss_rec.item() * images.size(0)

                test_total += images.size(0)

            ## LATENT DISTRIBUTION ##
            latent_images = self.actual_model.sample(len(images_first), labels_first, device=self.device, latents=True)
            latent_images = self.gather_into_tensor(latent_images)[:32, ...]

            latent_images = latent_images.detach().cpu().numpy()

            if is_master():
                image_grid = make_grid(latent_images)
                wandb.log({task + '/Generated_Latents': [
                    wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})

                latent_mean, latent_std = latent_images.mean(), latent_images.std()
                latent_min, latent_max = latent_images.min(), latent_images.max()
                latent_stats = {task + "/Latent_Mean": latent_mean, task + "/Latent_Std": latent_std,
                                task + "/Latent_Min": latent_min, task + "/Latent_Max": latent_max}

                for metric_key, metric_value in latent_stats.items():
                    wandb.log({metric_key: metric_value, '_step_': epoch})

            ## IMAGE DISTRIBUTION ##
            generate_images = self.actual_model.sample(len(images_first), labels_first, device=self.device, latents=False)
            generate_images = self.gather_into_tensor(generate_images)[:32, ...]

            generate_images = generate_images.detach().cpu().numpy()
            if is_master():
                image_grid = make_grid(generate_images)
                wandb.log({task + '/Generated_Images': [
                    wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})

                generated_mean, generated_std = generate_images.mean(), generate_images.std()
                generated_stats = {task + "/Generated_Mean": generated_mean, task + "/Generated_Std": generated_std}

                for metric_key, metric_value in generated_stats.items():
                    wandb.log({metric_key: metric_value, '_step_': epoch})

        self.actual_model.train()

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            self.log_to_wandb({metric_name: metric_score, '_step_': epoch})

        self.log_to_wandb({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total

        if task == 'Val':
            if epoch_val_loss < self.min_val_loss and is_master():
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)


def scale_to_minus_one_to_one(x):
    return x * 2 - 1


def reverse_scale_to_zero_to_one(x):
    return (x + 1) * 0.5


def set_random_entries_to_minus_one(labels):
    if random() > 0.8:
        return labels, None

    B, L = labels.shape
    num_entries_to_set = torch.randint(0, L, (B,))
    mask = torch.zeros_like(labels, dtype=torch.bool)

    for i in range(B):
        indices_to_set = torch.randperm(L)[:num_entries_to_set[i]]
        mask[i, indices_to_set] = True

    labels[mask] = -1
    return labels, mask
