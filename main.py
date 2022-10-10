import hydra

from conf.configs.utils import setup_hydra
from theory2practice.utils import setup_wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """
    Instantiate configurations and run experiments.
    """

    cfg = hydra.utils.instantiate(cfg, _convert_="all")
    cfg["runner"].run(cfg["specs"])


if __name__ == "__main__":
    setup_wandb()
    setup_hydra()
    main()
