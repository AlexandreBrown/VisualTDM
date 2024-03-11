# LEAP
Implementation of Goal-Conditioned Deep Reinforcement Learning approach called Latent Embeddings for Abstracted Planning (LEAP)  

Original paper : https://arxiv.org/abs/1911.08453  

<img src="https://i.ibb.co/ZV9cGjK/2024-03-11-12-28.png" height="300px">



# Getting Started
To get started you can use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).  
PS: If you use `conda` just replace `micromamba` with `conda` for the commands.

1. Create an empty environment.
    ```python
    micromamba create -n leap
    ```  
2. Activate the environment.
    ```python
    micromamba activate leap
    ```
3. Install `python` and `pip`
    ```python
    micromamba install -c conda-forge python=3.10 pip
    ```
4. Install the project dependencies.
    ```python
    pip3 install -r requirements.txt
    ```
# Explore Environment / Verify Installatin
You can explore an environment to verify that your setup is correct.
1. ```
   python explore_env.py env.name=AntMaze_UMaze-v4
   ```
2. The exploration of the environment will log a short video in the `outputs/` folder showing random actions in the environment.  
<img src="https://i.ibb.co/dBdQkkQ/exploration-example.png" height="200px">  

# End-To-End Training
## Train VAE  
1. ```
   python train_vae.py
   ```