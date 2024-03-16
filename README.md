# LEAP
Implementation of Goal-Conditioned Deep Reinforcement Learning approach called Latent Embeddings for Abstracted Planning (LEAP)  

Original paper : https://arxiv.org/abs/1911.08453  

<img src="https://i.ibb.co/ZV9cGjK/2024-03-11-12-28.png" height="300px">



# Getting Started
To get started you can use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).  
PS: If you use `conda` just replace `micromamba` with `conda` for the commands.

1. Create an empty environment.
    ```bash
    micromamba create -n leap
    ```  
1. Activate the environment.
    ```bash
    micromamba activate leap
    ```
1. Install `python` and `pip`
    ```bash
    micromamba install -c conda-forge python=3.10 pip
    ```
1. Install OpenGL dependencies for MuJoCo  
   ```bash
    conda install -c conda-forge glew
    conda install -c conda-forge mesalib
    conda install -c anaconda mesa-libgl-cos6-x86_64
    conda install -c menpo glfw3
   ```
1. Install the project dependencies.
    ```bash
    pip3 install -r requirements.txt
    ```
# Explore Environment / Verify Installatin
You can explore an environment to verify that your setup is correct.
1. ```bash
   python explore_env.py env.name=AntMaze_UMaze-v4
   ```
2. The exploration of the environment will log a short video in the `outputs/` folder showing random actions in the environment.  
<img src="https://i.ibb.co/dBdQkkQ/exploration-example.png" height="200px">  

# End-To-End Training
## Train VAE  
1. ```bash
   PYTHONPATH=./src python train_vae.py
   ```