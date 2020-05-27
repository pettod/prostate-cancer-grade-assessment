# Prostate Cancer Grade Assessment

Kaggle competition: Prostate cANcer graDe Assessment (PANDA) Challenge <https://www.kaggle.com/c/prostate-cancer-grade-assessment>

## Installation

During the development, Python 3.6 was used. Setup the project with the following 5 steps. There are commands on how to do the installation on Ubuntu (please be aware what they will do).

1. Install [Openslide](https://openslide.org/download/)

1. Install Python libraries

1. Download dataset

1. Move dataset into folder: `../input/`. You can also put the dataset to another hard drive and create symbolic link to it, add the symbolic link into `../input/`. The code will handle that.

1. Extract dataset

```shell
sudo apt-get install openslide-tools
pip install -e .
kaggle competitions download -c prostate-cancer-grade-assessment
mv prostate-cancer-grade-assessment.zip ../input/
cd ../input/
unzip prostate-cancer-grade-assessment.zip
```

**Note**: If there are issues with installing openslide from requirements.txt, remove the line, then run `pip install -e .` and install openslide manually.

## Running the Code

None of the following are required. Here is short a description what scripts the project has. Some of the constant values may needed to be changed in the beginning of the each file you run.

1. Split data into train and validation: `python scripts/split_train_and_valid_data.py`

1. Create cropped data for faster data generation during the training time: `python scripts/create_cropped_dataset.py`

1. Train Keras model: `python train_keras.py`

1. Train Pytorch model: `python train_pytorch.py`
