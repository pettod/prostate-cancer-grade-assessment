import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import glob
import numpy as np
import os
import time
from tqdm import tqdm, trange

# Project files
from src.pytorch.callbacks import EarlyStopping, CsvLogger
from src.pytorch.network import Net
from src.pytorch.utils import computeMetrics, computeLoss, plotLearningCurve
from src.image_generator import DataGenerator

# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "patches/train_256_4x4_mixed_res")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")
VALID_X_DIR = os.path.join(ROOT, "patches/valid_256_4x4_mixed_res")
VALID_Y_DIR = os.path.join(ROOT, "valid.csv")

# Model parameters
LOAD_MODEL = False
MODEL_PATH = None
BATCH_SIZE = 32
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 1e-4
NUMBER_OF_VALIDATION_ROUNDS_PER_EPOCH = 10

PROGRAM_TIME_STAMP = time.strftime("%Y-%m-%d_%H%M%S")


class Train():
    def __init__(self, device):
        self.device = device
        self.model_root = "models/pytorch"
        self.model = self.loadModel()
        save_model_directory = os.path.join(
            self.model_root, PROGRAM_TIME_STAMP)

        # Define optimizer and callbacks
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", 0.3, 3, min_lr=1e-8)
        self.early_stopping = EarlyStopping(save_model_directory, PATIENCE)
        self.csv_logger = CsvLogger(save_model_directory)
        self.current_epoch_metrics = {}

        # Define train and validation batch generators
        train_generator = DataGenerator(
            TRAIN_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
            normalize=True, rotate=True, add_gaussian_blur=True)
        self.train_batch_generator = train_generator.trainImagesAndLabels(
            TRAIN_Y_DIR, categorical_labels=False)
        self.number_of_train_batches = train_generator.numberOfBatchesPerEpoch()
        valid_generator = DataGenerator(
            VALID_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
            normalize=True)
        self.valid_batch_generator = valid_generator.trainImagesAndLabels(
            VALID_Y_DIR, categorical_labels=False)
        self.number_of_valid_batches = valid_generator.numberOfBatchesPerEpoch()
        self.validation_round_indices = np.linspace(
            int(self.number_of_train_batches /
            NUMBER_OF_VALIDATION_ROUNDS_PER_EPOCH),
            self.number_of_train_batches - 1,
            NUMBER_OF_VALIDATION_ROUNDS_PER_EPOCH, dtype=np.int)

    def loadModel(self):
        if LOAD_MODEL:

            # Load latest model
            if MODEL_PATH is None:
                model_name = sorted(glob.glob(os.path.join(
                    self.model_root, *['*', "*.pt"])))[-1]
            else:
                if type(MODEL_PATH) == int:
                    model_name = sorted(glob.glob(os.path.join(
                        self.model_root, *['*', "*.pt"])))[MODEL_PATH]
                else:
                    model_name = MODEL_PATH
            model = nn.DataParallel(Net()).to(self.device)
            model.load_state_dict(torch.load(model_name))
            model.eval()
            print("Loaded model: {}".format(model_name))
        else:
            model = nn.DataParallel(Net()).to(self.device)
        print("{:,} model parameters".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return model

    def validationRound(self):
        epoch_loss = 0
        epoch_kappa = 0
        epoch_accuracy = 0

        # Load tensor batch
        for i in range(self.number_of_valid_batches):
            X, y = next(self.valid_batch_generator)
            X = torch.from_numpy(np.moveaxis(X, -1, 1)).to(self.device)
            y = torch.tensor(y, dtype=torch.long).to(self.device)
            output = self.model(X)
            loss = computeLoss(output, y)
            batch_kappa, batch_accuracy = computeMetrics(y, output)
            epoch_kappa += (batch_kappa - epoch_kappa) / (i+1)
            epoch_accuracy += (batch_accuracy - epoch_accuracy) / (i+1)
            epoch_loss += (loss.item() - epoch_loss) / (i+1)
        print((
            "\n\n Valid loss: {:9.7f}. Kappa: {:4.3f}. " +
            "Accuracy: {:4.3f}").format(
                epoch_loss, epoch_kappa, epoch_accuracy))
        self.current_epoch_metrics["validation_loss"] = epoch_loss
        self.current_epoch_metrics["validation_kappa"] = epoch_kappa
        self.current_epoch_metrics["validation_accuracy"] = epoch_accuracy
        self.current_epoch_metrics["learning_rate"] = \
            self.optimizer.param_groups[0]["lr"]
        self.csv_logger.__call__(self.current_epoch_metrics)
        self.early_stopping.__call__(epoch_loss, self.model)
        self.scheduler.step(epoch_loss)

    def train(self):
        # Run epochs
        for epoch in range(EPOCHS):
            if self.early_stopping.isEarlyStop():
                print("Early stop")
                break
            progress_bar = trange(self.number_of_train_batches, leave=True)
            progress_bar.set_description(
                " Epoch {}/{}".format(epoch+1, EPOCHS))
            epoch_loss = 0
            epoch_kappa = 0
            epoch_accuracy = 0

            # Run batches
            for i in progress_bar:

                # Run validation data before last batch
                if i in self.validation_round_indices:
                    self.current_epoch_metrics["epoch"] = epoch + 1
                    self.current_epoch_metrics["training_loss"] = epoch_loss
                    self.current_epoch_metrics["training_kappa"] = epoch_kappa
                    self.current_epoch_metrics["training_accuracy"] = \
                        epoch_accuracy
                    with torch.no_grad():
                        self.validationRound()

                # Load tensor batch
                X, y = next(self.train_batch_generator)
                X = torch.tensor(np.moveaxis(X, -1, 1)).to(self.device)
                y = torch.tensor(y).to(self.device)

                # Feed forward and backpropagation
                self.model.zero_grad()
                output = self.model(X)
                loss = computeLoss(output, y)
                loss.backward()
                self.optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    batch_kappa, batch_accuracy = computeMetrics(y, output)
                    epoch_kappa += (batch_kappa - epoch_kappa) / (i+1)
                    epoch_accuracy += (batch_accuracy - epoch_accuracy) / (i+1)
                    epoch_loss += (loss.item() - epoch_loss) / (i+1)
                    progress_bar.display(
                        " Train loss: {:9.7f}. Kappa: {:4.3f}. Accuracy: {:4.3f}".format(
                            epoch_loss, epoch_kappa, epoch_accuracy), 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Running on CPU\n\n\n\n")

    train = Train(device)
    train.train()
    plotLearningCurve()


main()
