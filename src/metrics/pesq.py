import torch
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq_score,
)

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import align_sources


class PESQ(BaseMetric):
    def __init__(self, pit=False, *args, **kwargs):
        """
        Enable best-permutation matching with pit=True; otherwise keep source order.
        """
        super().__init__(*args, **kwargs)
        self.pit = pit

    def __call__(
        self, pred_sources, source_audios, audio_lengths, sample_rates, **kwargs
    ) -> float:
        """
        Return mean PESQ across examples and sources as a Python float.

        Accepts tensors [B, S, T] or gathered lists of [S, T] tensors.
        audio_lengths provides B valid lengths; padding is cropped per example.
        """
        scores = []
        for pred, target, length, fs in zip(
            pred_sources, source_audios, audio_lengths, sample_rates
        ):
            length = int(length.item())

            pred = pred[..., :length]
            target = target[..., :length]
            pred = align_sources(pred, target, pit=self.pit)
            fs = int(fs)
            score = pesq_score(pred, target, fs=fs, mode="wb")

            scores.append(score)

        return torch.stack(scores).mean().item()
