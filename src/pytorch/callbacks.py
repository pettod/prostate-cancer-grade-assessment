import torch
import numpy as np
import os
import pandas as pd


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
            self, save_model_directory, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_directory = save_model_directory

        # Create folders if do not exist
        if not os.path.isdir(save_model_directory):
            os.makedirs(save_model_directory)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            torch.save(model.state_dict(), "{}/model_last.pt".format(
                self.save_directory))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), "{}/model.pt".format(
            self.save_directory))
        self.val_loss_min = val_loss

    def isEarlyStop(self):
        return self.early_stop


class CsvLogger:
    def __init__(self, save_model_directory):
        self.logs_file_path = os.path.join(
            save_model_directory, "logs.csv")

    def __call__(self, loss_and_metrics):
        # Create CSV file
        new_data_frame = pd.DataFrame(loss_and_metrics, index=[0])
        if not os.path.isfile(self.logs_file_path):
            new_data_frame.to_csv(
                self.logs_file_path, header=True, index=False)
        else:
            with open(self.logs_file_path, 'a') as old_data_frame:
                new_data_frame.to_csv(
                    old_data_frame, header=False, index=False)
