# CLAR: Contrastive Learning of Auditory Representations

Codebase for CLAR a contrastive learning framework for auditory representations. In this repository, we only have PyTorch implementation. Please follow the steps below to ensure that the code work correctly.

# Install Python requirements

Before running the scripts, install all the python libraries:

```
pip install -r requirements.txt
```
 
Additionally, we pre-installed `cuda/10.1`, `cudnn/7.6.5`, and `nccl/2.5.6`. Instructions can be found here https://developer.nvidia.com/cuda-toolkit-archive
 
# Datasets

Most adopted dataset in the paper is [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz). This is a subset of the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). Before we utilize the dataset, we convert it to `lmdb` format for fast loading. If you would like to convert your own dataset to `lmdb` format, you could use `DatasetConverter.py`. Alternatively, you can adjust the method `load_datasets` in the code to use any dataset using the PyTorch framework.

We have provided the test data in `lmdb` format for SC09 for testing purposes. You can download it from https://drive.google.com/file/d/17TQvtKf1M3Mbm46uzd9SfXVBlrdB-Y9l/view

Once you have the lmdb files, create a directory named `data` and put those files in there.

# Training

There are 3 scripts for different methodology used in the paper: `main_supervised.py`, `main_unsupervised.py` and `semi_supervised.py`

To start training with default options, this command will do:

```
python main_supervised.py
```

The command above will train a 1D version of ResNet18. To change that, you can replace the name:

```
python main_supervised.py --model_name='2d'
```

As long as we have `2d` in the name, the model will use the 2D version of the model. If you would like to change the dataset used, you will need to replace the name:

```
python main_supervised.py --model_name='2d' --dataset='sc09'
```

This expect that you have `train_sc09.lmdb`, `valid_sc09.lmdb` and `test_sc09.lmdb` in your data folder.

To add augmentations to the training, you will need to pass parameters that reflect a given augmentation. The set of augmentations with the best results are as follows:

```
python main_supervised.py --model_name='2d' --dataset='sc09' --tm=True --fd=True
```

# Testing

To get the test accuracy, all we need to set `train` parameter to `False`:

 ```
python main_supervised.py --model_name='2d' --dataset='sc09' --tm=True --fd=True --train=False
```

Make sure that you have the saved model in the `./model` directory. 

Please note that depending on the augmentations and dataset used the model name will change. For instance, for the above command the model name will be `2d_100_fd_tm`. This was done to save models for different augmentations without losing track. Hence, the script will look for the saved model at `./models/2d_100_fd_tm/sc09/`. The `100` in the number indicate that we used 100% of the labeled data to train the model.

# Saved Models

Models for the final results can be downloaded from https://drive.google.com/drive/folders/1GA-HCBXlRuOCWHQ-_A_Ei265OEek2kms?usp=sharing

Please make sure to maintain the folder structure.