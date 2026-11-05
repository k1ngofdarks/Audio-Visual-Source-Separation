from itertools import permutations

import torch
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
)


def si_snr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return SI-SNR in dB for matching tensors (..., T), with output shape (...,).
    """
    return scale_invariant_signal_noise_ratio(preds, target)


def pit_si_snr(
    preds: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Find the permutation with the highest mean SI-SNR for one example [S, T].

    Inputs must already exclude padding. Returns differentiable scores [S]
    in target order and a tuple perm: preds[list(perm)] matches target.
    """
    S = preds.shape[0]

    perms = list(permutations(range(S)))
    si_snr_values = []
    for perm in perms:
        permuted_preds = preds[list(perm)]
        si_snr_value = si_snr(permuted_preds, target)
        si_snr_values.append(si_snr_value)

    scores = torch.stack(si_snr_values)
    index = scores.mean(dim=1).argmax()

    index = index.item()
    return scores[index], perms[index]


def align_sources(
    preds: torch.Tensor, target: torch.Tensor, pit: bool = False
) -> torch.Tensor:
    """Align predictions [S, T] to reference order using mean SI-SNR.

    Inputs must already exclude padding. With pit=False, keep the original
    order. References stay unchanged; reordered predictions retain gradients.
    """
    if pit:
        _, permutation = pit_si_snr(preds, target)
        preds = preds[list(permutation)]
    return preds


def sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return SDR in dB for matching tensors (..., T), with output shape (...,).
    """
    return signal_distortion_ratio(preds, target)
