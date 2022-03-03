from omegaconf import DictConfig, OmegaConf
import hydra
import tensorboard
import numpy as np

np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="configs", config_name="defaults")
def Unsup_Syn(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


    experiment = hydra.utils.instantiate(cfg.Experiment)

    model = hydra.utils.instantiate(cfg.Model, in_channels = cfg.Experiment.in_channels, out_channels = cfg.Experiment.out_channels).to(device)

    regulariser = hydra.utils.instantiate(cfg.Regulariser)

    optimizer 

if __name__ == "__main__":
    Unsup_Syn()