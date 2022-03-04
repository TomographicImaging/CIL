import imp
from os import device_encoding
from turtle import pen
from omegaconf import DictConfig, OmegaConf
import hydra
from tensorboardX import SummaryWriter
import numpy as np
from Online_RNA import Online_RNA
import os

np.random.seed(0)

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    device = 'gpu'
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    device = 'cpu'
    print('No Nvidia GPU in system!')

@hydra.main(config_path="configs", config_name="defaults")
def RNA(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=os.getcwd())

    experiment = hydra.utils.instantiate(cfg.Experiment, device=device)

    hyperparameters = hydra.utils.instantiate(cfg.Hyperparameters)

    function = hydra.utils.instantiate(cfg.Function, experiment = experiment)

    algorithm = hydra.utils.instantiate(cfg.Algorithm, experiment = experiment, function = function)

    if hyperparameters.RNA == True:
        RNA_algo = Online_RNA(algo = algorithm.get_Algo(), RNA_reg_param=hyperparameters.RNA_reg, N=hyperparameters.N, beta=hyperparameters.Beta)

        RNA_algo.run(hyperparameters.Epochs)

        for i in range(hyperparameters.Epochs):
            writer.add_scalar('objective', RNA_algo.objective[i], i)
    else:
        Algo = algorithm.get_Algo()
        Algo.run(hyperparameters.Epochs)
        for i in range(hyperparameters.Epochs):
            writer.add_scalar('objective', Algo.objective[i], i)



if __name__ == "__main__":
    RNA()