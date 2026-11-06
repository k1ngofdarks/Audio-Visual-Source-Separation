from transformers import AutoConfig, AutoModel

from src.model.baseline_model import BaselineConfig, BaselineModel
from src.model.video_encoder.lipreading_model import LipreadingConfig, LipreadingModel


def register_models():
    # Register all models and their configs
    AutoConfig.register(BaselineConfig.model_type, BaselineConfig)
    AutoModel.register(BaselineConfig, BaselineModel)

    AutoConfig.register(LipreadingConfig.model_type, LipreadingConfig)
    AutoModel.register(LipreadingConfig, LipreadingModel)
