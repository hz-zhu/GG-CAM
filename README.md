# GG-CAM

**This repo contains python codes used for "Gaze-Guided Class Activation Mapping: Leveraging Human Attention for Network Attention in Chest X-rays Classification", Hongzhi Zhu et al., submitted to International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2022.**

- Folder _Data_ _Processing_, contains code used for the pre-processing and splitting of raw dataset.
- Folder _EffNet+GG-CAM_, contains code used for the training and testing of EffNet+GG-CAM.
- Folder _ResNet+GG-CAM_, contains code used for the training and testing of ResNet+GG-CAM.

## _Data_ _Processing_ folder

data_processing.py is the main file for data pre-processing.\
Python 3.8.5 is used. Packages required for data_processing.py are:
- PyTorch 1.8.0
- pandas 1.1.0
- matplotlib 3.3.1
- numpy 1.19.2
- opencv 4.0.1

To run the code, simply execute data_processing.py through “python data_processing.py” in command line or with other Python IDEs. After execution, new folders containing per-processed and split datasets ready for training will be created in the execution directory. The raw dataset is partitioned into training (70%), validation (10%) and testing (20%) subsets. Seeding (value 0) is used for reproducibility.

## _EffNet+GG-CAM_ folder

ENetCAM.py is the main file for the training and evaluation of EffNet+GG-CAM.\
Python 3.8.5 is used. Packages required for ENetCAM.py are:
- PyTorch 1.8.0
- torchvision 0.9.0
- tensorboard 2.3.0
- pandas 1.1.0
- matplotlib 3.3.1
- numpy 1.19.2
- scipy 1.5.2
- sklearn 0.23.2

To run the code, simply execute ENetCAM.py through “python ENetCAM.py” in command line or with other Python IDEs. After execution, a new folder “run” will be created (if not already exists) in the execution directory. For each execution of ENetCAM.py, a new folder with a random name will be created inside folder “run” for the temporary storage of network parameters as well as recoding training details and evaluation results. The following lists the file details inside the folder:
- params_NetLearn_.txt, records parameters and data related to network training.
- params_EffCAMNet_.txt, records hyper-parameters for the network.
- QuickHelper_summary.txt, records other data during execution.
- ENetCAM_S_XXXX/classification_report.json, records classification evaluation metrics, and XXXXX are random characters generated during run time.
- ENetCAM_S_XXXX/classification_results.csv, records raw classification output for each test image.
- ENetCAM_S_XXXX/NET.pt, stores the parameters for the best performing network during or after training.
- ENetCAM_S_XXXX/training_process.png, visualizes the change of learning rate, losses, and validation metrics during the training process.

During execution, ENetCAM.py will also print training details and evaluation results to the console.
