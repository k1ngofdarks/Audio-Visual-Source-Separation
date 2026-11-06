from torch import Tensor, nn
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize


class VideoNormalize(nn.Module):
    """
    Video preprocessing class for downstream video encoding.
    """

    def __init__(self, output_size, mean, std):
        """
        Args:
            output_size (tuple[int, int] | int): size of each frame in the output video.
            mean (float): mean value used in normalization.
            std (float): std value used in normalization.
        """
        super().__init__()

        self.transform = Compose(
            [
                Normalize((0.0,), (255.0,)),
                CenterCrop(output_size),
                Normalize((mean,), (std,)),
            ]
        )

    def forward(self, video: Tensor):
        """
        Args:
            video (Tensor): input video tensor of shape (*, H, W).
        Returns:
            normalized_video (Tensor): normalized video tensor.
        """
        normalized_video = self.transform(video.float())
        return normalized_video
