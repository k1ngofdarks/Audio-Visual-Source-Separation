from torch import Tensor, nn
from torch.nn import SiLU
from transformers import PreTrainedConfig, PreTrainedModel

from src.model.video_encoder.resnet import ResNet


class LipreadingConfig(PreTrainedConfig):
    model_type = "lipreading_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LipreadingModel(PreTrainedModel):
    """
    Implements Lipreading Video Encoder
    https://arxiv.org/pdf/2209.01383.pdf
    """

    config_class = LipreadingConfig
    frontend_nout = 64

    def __init__(self, config):
        """
        Args:
            config (LipreadingConfig): configuration.
        """
        super().__init__(config)

        self.trunk = ResNet(layers=[2, 2, 2, 2])
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            SiLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.post_init()

    def forward(self, mouth_videos: Tensor):
        """
        Args:
            mouth_videos (Tensor): normalized mouth videos of shape (B, 1, T, H, W)
        Returns:
            video_embeddings (Tensor): video embeddings of shape (B, Tnew, E)
        """
        x = self.frontend3D(mouth_videos)
        batch_size, _, Tnew, _, _ = x.shape
        x = x.transpose(1, 2)
        x = x.flatten(end_dim=1)
        x = self.trunk(x)
        x = x.view(batch_size, Tnew, x.shape[1])

        return x
