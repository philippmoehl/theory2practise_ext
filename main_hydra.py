import hydra
from hydra.utils import get_method, get_class
from omegaconf import DictConfig, OmegaConf

from theory2practice import utils

OmegaConf.register_new_resolver(name="get_method",
                                resolver=lambda cls: get_method(cls))
OmegaConf.register_new_resolver(name="get_cls",
                                resolver=lambda cls: get_class(cls))
OmegaConf.register_new_resolver(name="get_dtype",
                                resolver=lambda cls: utils.get_dtype(cls))
OmegaConf.register_new_resolver(name="max",
                                resolver=lambda n: int(max(n, 1)))


@hydra.main(version_base=None, config_path="conf/test",
            config_name="config_umbrella")
def main(cfg: DictConfig):
    utils.setup()
    runner = hydra.utils.instantiate(cfg.runner, _convert_="partial")
    exp = hydra.utils.instantiate(cfg.experiment, _convert_="partial")

    runner.run([exp])


if __name__ == "__main__":
    main()
