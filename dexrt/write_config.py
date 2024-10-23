import yaml
from dexrt.config_schemas import DexrtConfig, DexrtRayConfig


def write_config(conf: DexrtConfig | DexrtRayConfig, path="dexrt.yaml"):
    """Write the pydantic config schema provided to a yaml file at path.
    Paths internal to the config aren't adjusted.
    """
    data = conf.model_dump()
    with open(path, "w") as f:
        yaml.dump(data, f)
