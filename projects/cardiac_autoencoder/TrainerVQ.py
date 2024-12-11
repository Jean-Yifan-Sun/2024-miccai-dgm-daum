from core.Trainer import Trainer
from time import time
import wandb
import logging
import torch
from core.Main import get_rank
from torchinfo import summary
from core.Main import is_master

from dl_utils.image_utils import make_comparison_grid, make_grid
from model_zoo.autoencoder import Action
from model_zoo.patch_gan_discriminator import PatchGANDiscriminator

class PTrainer(Trainer):
    def __init__(self, training_params, training_mode, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, training_mode, model, data, device, log_wandb)
        image_size = (3, ) + training_params['input_size']
        logging.info(f"[Trainer::summary]: Model in: {image_size}")
        logging.info(f"[Trainer::summary]: {str(summary(self.actual_model, image_size))}")


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
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        self.early_stop = False

        self.patch_gan_discriminator = PatchGANDiscriminator(input_size=self.training_params['input_size'][-3:],
                                                             hidden_layers=self.training_params['patchgan_channels'],
                                                             across_channels=self.training_params[
                                                                 'across_channels']).to(self.device)

        self.patch_gan_discriminator = self.patch_gan_discriminator.to(self.device)
        self.criterion_discriminator = torch.nn.BCEWithLogitsLoss()
        self.discriminator_optimizer = torch.optim.Adam(self.patch_gan_discriminator.parameters(),
                                                        lr=self.training_params['patchgan_lr'])
        self.barrier("pre training loop")

        epoch_losses = []
        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            start_time = time()
            (batch_loss, batch_loss_rec, batch_loss_vq,
             batch_loss_pl, batch_loss_adv, batch_loss_disc, count_images) = 0, 0, 0, 0, 0, 0, 0
            logging.info(f"[Trainer::summary]: (rank {get_rank()}) Epoch: {epoch}  Dataloader size: {len(self.train_ds)}")
            batch_count = 0
            for idx, data in enumerate(self.train_ds):
                batch_count += 1
                # continue
                logging.info(f"[Trainer::summary]: (rank {get_rank()}) idx: {idx}")

                # Input
                batch = [b.to(self.device) for b in data]
                # process batch data
                images, labels = batch

                ### Train VAE (Generator) ###
                self.optimizer.zero_grad()

                x_recon, loss_vq = self.model(images)
                
                # logging.info(f'image size{images.shape},x_recon size{x_recon.shape}')
                # image sizetorch.Size([32, 96, 96, 13]),x_recon sizetorch.Size([32, 96, 96, 12])
                
                # Calculate normal losses
                loss_rec = self.criterion_rec(x_recon, images) * 1000
                loss_pl = self.criterion_PL(x_recon, images) * 100
                loss_vq = loss_vq.mean()

                # Calculate adversarial loss
                fake_labels = torch.ones(images.size(0), 1, device=self.device)  # Target labels as real
                fake_ground_truth = self.patch_gan_discriminator.generate_ground_truth(fake_labels, self.device)
                discriminator_output = self.patch_gan_discriminator(x_recon)  # Pass generated images
                loss_adv = self.criterion_discriminator(discriminator_output, fake_ground_truth) * 50

                loss = loss_rec + loss_vq + loss_adv + loss_pl
                loss.backward()
                self.optimizer.step()

                ### Train Discriminator ###
                self.discriminator_optimizer.zero_grad()

                # Train with real images
                real_labels = torch.ones(images.size(0), 1, device=self.device)
                real_ground_truth = self.patch_gan_discriminator.generate_ground_truth(real_labels, self.device)
                real_output = self.patch_gan_discriminator(images)
                loss_real = self.criterion_discriminator(real_output, real_ground_truth)

                # Train with fake images
                fake_labels = torch.zeros(images.size(0), 1, device=self.device)
                fake_ground_truth = self.patch_gan_discriminator.generate_ground_truth(fake_labels, self.device)
                fake_output = self.patch_gan_discriminator(x_recon.detach())  # Detach to avoid backprop to generator
                loss_fake = self.criterion_discriminator(fake_output, fake_ground_truth)

                # Total discriminator loss
                loss_discriminator = ((loss_real + loss_fake) / 2) / 2
                loss_discriminator.backward()
                self.discriminator_optimizer.step()

                # Log
                batch_loss += loss.item() * images.size(0)
                batch_loss_rec += loss_rec.item() * images.size(0)
                batch_loss_vq += loss_vq.item() * images.size(0)
                batch_loss_pl += loss_pl.item() * images.size(0)
                batch_loss_adv += loss_adv.item() * images.size(0)
                batch_loss_disc += loss_discriminator.item() * images.size(0)

                count_images += images.size(0)
            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            logging.info('Epoch: {} \tVAE Training Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            self.log_to_wandb({"Train/Loss_": epoch_loss,
                       "Train/Loss_rec_": batch_loss_rec / count_images,
                       "Train/Loss_vq_": batch_loss_vq / count_images,
                       "Train/Loss_pl_": batch_loss_pl / count_images,
                       "Train/Loss_adv_": batch_loss_adv / count_images,
                       "Train/Loss_disc_": batch_loss_disc / count_images,
                       '_step_': epoch})

            if self.training_mode.is_DP():
                DELTA = 1e-5
                epsilon = self.privacy_engine.get_epsilon(DELTA)
                logging.info(f"Differential privacy (ε = {epsilon:.2f}, δ = {DELTA})")

            # Plot train reconstruction
            rec = x_recon.detach().cpu()[0:5].numpy()
            img = images.detach().cpu()[0:5].numpy()
            grid_image = make_comparison_grid(img, rec)
            self.log_to_wandb({"Train/Reconstruction_": [
                wandb.Image(grid_image, caption="Iteration_" + str(epoch))]})

            if is_master():
                # Save latest model
                torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                               , 'epoch': epoch}, self.client_path + '/latest_model.pt')

                torch.save({'model_weights': self.actual_model.state_dict()}, self.client_path + '/latest_actual_model.pt')

            self.barrier("epoch end")
            # Run validation
            if epoch % 5 == 0 and epoch > 0:
                self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.actual_model.state_dict(), self.optimizer.state_dict()


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
        self.model.eval()

        metrics = {
            task + '/Loss_rec': 0,
            task + '/Loss_mse': 0,
            task + '/Loss_pl': 0,
            task + '/Loss_vq': 0,
            task + '/Reconstruction_mean': 0,
            task + '/Reconstruction_std': 0,
            task + '/Z_mean': 0,
            task + '/Z_std': 0,

            task + '/Z_n0_mean': 0,
            task + '/Z_n0_std': 0,
            task + '/Z_0_mean': 0,
            task + '/Z_0_std': 0,
        }
        test_total = 0
        with torch.no_grad():
            for idx, data in enumerate(test_data):
                logging.info(f"[Trainer::summary]: (rank {get_rank()}) Val idx: {idx}")

                batch = [b.to(self.device) for b in data]
                images, labels = batch
                # Forward pass
                x_recon, loss_vq = self.model(images)
                loss_vq = loss_vq.mean()
                z = self.model(images, Action.ENCODE).squeeze(1)

                images = self.gather_into_tensor(images)
                x_recon = self.gather_into_tensor(x_recon)
                z = self.gather_into_tensor(z)

                loss_rec = self.criterion_rec(x_recon, images) * 100
                loss_mse = self.criterion_MSE(x_recon, images)
                loss_pl = self.criterion_PL(x_recon, images) * 100

                metrics[task + '/Loss_rec'] += loss_rec.item() * images.size(0)
                metrics[task + '/Loss_mse'] += loss_mse.item() * images.size(0)
                metrics[task + '/Loss_pl'] += loss_pl.item() * images.size(0)
                metrics[task + '/Loss_vq'] += loss_vq.item() * images.size(0)

                metrics[task + '/Reconstruction_mean'] += x_recon.mean().item() * images.size(0)
                metrics[task + '/Reconstruction_std'] += x_recon.std().item() * images.size(0)
                metrics[task + '/Z_mean'] += z.mean().item() * images.size(0)
                metrics[task + '/Z_std'] += z.std().item() * images.size(0)

                non_zero_means, non_zero_stds = [], []
                zero_means, zero_stds = [], []
                # Loop over the batch dimension
                for i in range(images.size(0)):  # B
                    # Identify non-zero slices in the images
                    # We sum over the height (H) and width (W) dimensions and check if the result is not zero
                    non_zero_slices_mask = images[i].sum(dim=(1, 2)) != 0

                    # Use the mask to select the corresponding non-zero slices in z
                    z_non_zero_slices = z[i][non_zero_slices_mask]

                    if z_non_zero_slices.numel() > 0:  # Check if there are any non-zero slices
                        # Calculate mean and std for the non-zero slices in z
                        mean_z_non_zero = z_non_zero_slices.mean().item()
                        std_z_non_zero = z_non_zero_slices.std().item()

                        # Append the results to the lists
                        non_zero_means.append(mean_z_non_zero)
                        non_zero_stds.append(std_z_non_zero)

                    z_zero_slices = z[i][~non_zero_slices_mask]
                    if z_zero_slices.numel() > 0:  # Check if there are any non-zero slices
                        # Calculate mean and std for the non-zero slices in z
                        mean_z_zero = z_zero_slices.mean().item()
                        std_z_zero = z_zero_slices.std().item()

                        # Append the results to the lists
                        zero_means.append(mean_z_zero)
                        zero_stds.append(std_z_zero)

                metrics[task + '/Z_n0_mean'] += sum(non_zero_means)
                metrics[task + '/Z_n0_std'] += sum(non_zero_stds)

                metrics[task + '/Z_0_mean'] += sum(zero_means)
                metrics[task + '/Z_0_std'] += sum(zero_stds)

                test_total += images.size(0)

        rec = x_recon.detach().cpu()[0:5].numpy()
        img = images.detach().cpu()[0:5].numpy()

        grid_image = make_comparison_grid(img, rec)

        self.log_to_wandb({task + '/Reconstruction': [
            wandb.Image(grid_image, caption="Iteration_" + str(epoch))]})

        # Plot latent space
        z = z.detach().cpu()[0:5].numpy()
        z = make_grid(z)

        self.log_to_wandb({task + '/Latent_space': [
            wandb.Image(z, caption="Iteration_" + str(epoch))]})


        for metric_key in metrics.keys():
            metric_name = str(metric_key)
            metric_score = metrics[metric_key] / test_total
            self.log_to_wandb({metric_name: metric_score, '_step_': epoch})
        self.log_to_wandb({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '/Loss_mse'] / test_total

        if epoch_val_loss < self.min_val_loss and is_master():
            self.min_val_loss = epoch_val_loss
            torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                       self.client_path + '/best_model.pt')
            torch.save({'model_weights': self.actual_model.state_dict()}, self.client_path + '/best_actual_model.pt')
            self.best_weights = model_weights
            self.best_opt_weights = opt_weights
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_val_loss)

        self.model.train()
        return metrics

