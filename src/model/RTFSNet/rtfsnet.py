from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig, PreTrainedModel

from src.model.video_encoder.lipreading_model import LipreadingConfig, LipreadingModel


@strict(accept_kwargs=True)
class RTFSNetConfig(PreTrainedConfig):
    model_type = "rtfsnet"

    def __init__(
        self,
        video_encoder_name_or_path="dla-project/lipreading",
        freeze_video_encoder=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.video_encoder_name_or_path = video_encoder_name_or_path
        self.freeze_video_encoder = freeze_video_encoder


class RTFSNet(PreTrainedModel):
    """
    RTFSNet model for audio-visual speech separation.
    based on this paper: https://arxiv.org/pdf/2309.17189
    """

    config_class = RTFSNetConfig

    def __init__(self, config):
        super().__init__(config)

        self.video_encoder = LipreadingModel(LipreadingConfig())

        self.audio_encoder = ...
        self.separation_net = ...
        self.audio_decoder = ...

        self.post_init()

        if not next(self.video_encoder.parameters()).is_meta:
            pretrained = LipreadingModel.from_pretrained(
                config.video_encoder_name_or_path
            )
            self.video_encoder.load_state_dict(pretrained.state_dict())

        if config.freeze_video_encoder:
            self.video_encoder.requires_grad_(False)
            self.video_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_video_encoder:
            self.video_encoder.requires_grad_(False)
            self.video_encoder.eval()
        return self

    def forward(self, mix_audios, mouth_videos, **batch):
        # mix_audios: [B, L_a]
        # mouth_videos: [B, S, L_v, H, W]

        return {"pred_sources": ...}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
