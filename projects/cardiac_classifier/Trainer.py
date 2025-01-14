from core.Trainer import Trainer
from time import time
import wandb
import logging
import torch
import numpy as np
from dl_utils.image_utils import make_comparison_grid, make_grid

class PTrainer3D(Trainer):
    def __init__(self, training_params, training_mode, model, data, device, log_wandb=True):
        super(PTrainer3D, self).__init__(training_params, training_mode, model, data, device, log_wandb)

        self.esed_dim = self.training_params['esed_dim']

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

        epoch_losses = []
        for epoch in range(self.training_params['nr_epochs']):
            start_time = time()
            (batch_loss, count_images) = 0, 0

            for idx, data in enumerate(self.train_ds):
                # Input
                batch = [b.to(self.device) for b in data]
                images, labels = batch
                # labels = labels[:, self.esed_dim]

                self.optimizer.zero_grad()

                labels_pred = self.model(images)
                # logging.info(f'labels_pred: {labels_pred.shape}, labels: {labels.shape}')
                loss = self.criterion_rec(labels_pred, labels.float())
                loss.backward()

                self.optimizer.step()

                # Log
                batch_loss += loss.item() * images.size(0)
                count_images += images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            logging.info('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss,
                       '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            if epoch % 10 == 0 and epoch>1:
                # logging.info(type(self.val_ds))
                self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights


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
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()

        metrics = {
            task + '/Loss_BCE': 0,
            task + '/Accuracy': 0,
        }
        test_total = 0
        # logging.info(len(test_data)) 
        with torch.no_grad():
            for data in test_data:
                # logging.info(f'val data: {data}')
                batch = [b.to(self.device) for b in data]
                images, labels = batch
                # labels = labels[:, self.esed_dim]

                # Forward pass
                labels_pred = self.test_model(images)

                loss_rec = self.criterion_rec(labels_pred, labels.float())

                metrics[task + '/Loss_BCE'] += loss_rec.item() * images.size(0)

                labels_pred = (labels_pred > 0).int()
                metrics[task + '/Accuracy'] += (1 - ((labels_pred - labels).cpu().abs().sum() / images.size(0)).item()) * images.size(0)
                logging.info(f'labels_pred :{labels_pred.shape}')
                test_total += images.size(0)

        logging.info(f'images: {images.shape}')
        images_es = images*(labels_pred == 0)
        images_es = images_es.detach().cpu().numpy()
        images_ed = images*(labels_pred == 1)
        images_ed = images_ed.detach().cpu().numpy()

        logging.info(f'images_ed: {images_ed.shape}')
        logging.info(f'images_es: {images_es.shape}')

        if len(images_es) > 0:
            image_grid = make_grid(images_es)
            wandb.log({task + '/Predicted_ES': [
                wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})

        if len(images_ed) > 0:
            image_grid = make_grid(images_ed)
            wandb.log({task + '/Predicted_ED': [
                wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})


        for metric_key in metrics.keys():
            metric_name = str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '/Loss_BCE'] / test_total

        if epoch_val_loss < self.min_val_loss:
            self.min_val_loss = epoch_val_loss
            torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                       self.client_path + '/best_model.pt')
            self.best_weights = model_weights
            self.best_opt_weights = opt_weights

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_val_loss)

        return metrics


class PTrainer2D(Trainer):
    def __init__(self, training_params, training_mode, model, data, device, log_wandb=True):
        super(PTrainer2D, self).__init__(training_params, training_mode, model, data, device, log_wandb)

        self.esed_dim = self.training_params['esed_dim']

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

        epoch_losses = []
        for epoch in range(self.training_params['nr_epochs']):
            start_time = time()
            (batch_loss, count_images) = 0, 0

            for idx, data in enumerate(self.train_ds):
                # Input
                batch = [b.to(self.device) for b in data]
                images, labels = batch
                # labels = labels[:, self.esed_dim]
                images = self.toSlices(images)
                labels = self.toSlices(labels)

                self.optimizer.zero_grad()

                labels_pred = self.model(images)
                # logging.info(f'labels_pred: {labels_pred.shape}, labels: {labels.shape}')
                loss = self.criterion_rec(labels_pred, labels.float())
                loss.backward()

                self.optimizer.step()

                # Log
                batch_loss += loss.item() * images.size(0)
                count_images += images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            logging.info('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss,
                       '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            if epoch % 10 == 0 and epoch>1:
                # logging.info(type(self.val_ds))
                self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

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
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()

        metrics = {
            task + '/Loss_BCE': 0,
            task + '/Accuracy': 0,
        }
        test_total = 0
        # logging.info(len(test_data))
        with torch.no_grad():
            for data in test_data:
                # logging.info(f'val data: {data}')
                batch = [b.to(self.device) for b in data]
                images, labels = batch
                # labels = labels[:, self.esed_dim]
                images = self.toSlices(images)
                labels = self.toSlices(labels)
                # Forward pass
                labels_pred = self.test_model(images)

                loss_rec = self.criterion_rec(labels_pred, labels.float())

                metrics[task + '/Loss_BCE'] += loss_rec.item() * images.size(0)

                images = self.unSlice(images)
                labels = self.unSlice(labels)
                labels_pred = self.unSlice(labels_pred)

                labels_pred = (labels_pred > 0).int()
                metrics[task + '/Accuracy'] += (1 - ((labels_pred - labels).cpu().abs().sum() / images.size(0)).item()) * images.size(0)
                logging.info(f'labels_pred :{labels_pred.shape}')
                test_total += images.size(0)

        logging.info(f'images: {images.shape}')
        images_es = images*(labels_pred == 0)
        images_es = images_es.detach().cpu().numpy()
        images_ed = images*(labels_pred == 1)
        images_ed = images_ed.detach().cpu().numpy()

        logging.info(f'images_ed: {images_ed.shape}')
        logging.info(f'images_es: {images_es.shape}')

        if len(images_es) > 0:
            image_grid = make_grid(images_es)
            wandb.log({task + '/Predicted_ES': [
                wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})

        if len(images_ed) > 0:
            image_grid = make_grid(images_ed)
            wandb.log({task + '/Predicted_ED': [
                wandb.Image(image_grid, caption="Iteration_" + str(epoch))]})


        for metric_key in metrics.keys():
            metric_name = str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '/Loss_BCE'] / test_total

        if epoch_val_loss < self.min_val_loss:
            self.min_val_loss = epoch_val_loss
            torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                       self.client_path + '/best_model.pt')
            self.best_weights = model_weights
            self.best_opt_weights = opt_weights

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_val_loss)

        return metrics



class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pth'):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): Whether to print updates.
            path (str): File path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when the validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
