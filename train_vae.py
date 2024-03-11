import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs/", config_name="vae")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
