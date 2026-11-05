import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import pit_sdr, sdr


class SDR(BaseMetric):
    def __init__(self, pit=False, *args, **kwargs):
        """
        Enable best-permutation matching with pit=True; otherwise keep source order.
        """
        super().__init__(*args, **kwargs)
        self.pit = pit

    def __call__(self, pred_sources, source_audios, audio_lengths, **kwargs) -> float:
        """
        Return mean SDR in dB across examples and sources as a Python float.

        Accepts tensors [B, S, T] or gathered lists of [S, T] tensors.
        audio_lengths provides B valid lengths; padding is cropped per example.
        """
        scores = []
        for pred, target, length in zip(pred_sources, source_audios, audio_lengths):
            length = int(length.item())

            pred = pred[..., :length]
            target = target[..., :length]

            if self.pit:
                score, best_permutation = pit_sdr(pred, target)
            else:
                score = sdr(pred, target)

            scores.append(score)

        return torch.stack(scores).mean().item()


class SDRi(BaseMetric):
    def __init__(self, pit=False, *args, **kwargs):
        """
        Enable PIT for predictions with pit=True; mixture scores use fixed order.
        """
        super().__init__(*args, **kwargs)
        self.pit = pit

    def __call__(
        self, pred_sources, source_audios, mix_audios, audio_lengths, **kwargs
    ) -> float:
        """
        Return mean SDR improvement over the mixture in dB as a Python float.

        Predictions/references are [B, S, T], mixtures are [B, T]; gathered
        lists also work. Crop by audio_lengths before scoring and averaging.
        """
        scores = []
        for pred, target, mix, length in zip(
            pred_sources, source_audios, mix_audios, audio_lengths
        ):
            length = int(length.item())

            pred = pred[..., :length]
            target = target[..., :length]
            mix = mix[..., :length].unsqueeze(0).expand_as(pred)

            if self.pit:
                score_pred, best_permutation = pit_sdr(pred, target)
            else:
                score_pred = sdr(pred, target)
            score_mix = sdr(mix, target)

            scores.append(score_pred - score_mix)

        return torch.stack(scores).mean().item()
