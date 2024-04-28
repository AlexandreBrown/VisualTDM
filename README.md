# Image-Based Temporal Difference Models
Implementation of Goal-Conditioned Deep Reinforcement Learning approach called Temporal Difference Models (TDMs) but applied to images using latent representation.

Original paper : https://arxiv.org/abs/1802.09081

My Report : See [IFT6163_RL_Project_Report.pdf](./IFT6163_RL_Project_Report.pdf)

Experiments: https://www.comet.com/alexandrebrown/visual-tdm/view/new/panels

# Getting Started
To get started you can use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).  
PS: If you use `conda` just replace `micromamba` with `conda` for the commands.

1. Create an empty environment.
    ```bash
    micromamba create -n visual-tdm
    ```  
1. Activate the environment.
    ```bash
    micromamba activate visual-tdm
    ```
1. Install `python` and `pip`
    ```bash
    micromamba install -c conda-forge python=3.10 pip
    ```
1. Install OpenGL dependencies for MuJoCo  
   ```bash
    micromamba install -c conda-forge glew
    micromamba install -c conda-forge mesalib
    micromamba install -c anaconda mesa-libgl-cos6-x86_64
    micromamba install -c menpo glfw3
   ```
1. Install the project dependencies.
    ```bash
    pip3 install -r requirements.txt
    ```
Note: Parts of this project requires a CometML account for logging metrics.

# Explore Environment / Verify Installatin
You can explore an environment to verify that your setup is correct.
1. ```bash
   PYTHONPATH=./src python explore_env.py env=antmaze_umaze
   ```
2. The exploration of the environment will log a short video in the `outputs/` folder showing random actions in the environment.  
<img src="https://i.ibb.co/dBdQkkQ/exploration-example.png" height="200px">  

# End-To-End Training
You can find the commands in the `.vscode/launch.json` file.  
## Generate VAE Dataset
1. ```bash
   PYTHONPATH=./src python generate_vae_dataset.py env=antmaze_umaze
   ```
## Train VAE  
1. Define the environment variables for [CometML](https://www.comet.com/site/) logging.  
    ```bash
    export COMET_ML_API_KEY=<YOUR_API_KEY>  
    export COMET_ML_PROJECT_NAME=<YOUR_PROJECT_NAME>
    export COMET_ML_WORKSPACE=<YOUR_WORKSPACE>
    ```
1. ```bash
   PYTHONPATH=./src python train_vae.py env=antmaze_umaze dataset.path=datasets/vae_dataset_PointMaze_UMaze-v3_65536.h5
   ```  
   Make sure to put your generated dataset under `datasets/` beforehand.  

## Train TDM  
1. ```bash
    PYTHONPATH=./src python train_tdm.py env=antmaze_umaze models.encoder_decoder.name=vae_best_model_pointmaze_umaze-v3
   ```

## Train TD3 (Baseline)  
1. ```bash
    PYTHONPATH=./src python train_td3.py env=antmaze_umaze
   ```

# Find Experiments Results  
All experiments were made publicly available along with the model weights : https://www.comet.com/alexandrebrown/visual-tdm/view/new/experiments  
Archived experiments that represented unsatisfying results were deleted.  
