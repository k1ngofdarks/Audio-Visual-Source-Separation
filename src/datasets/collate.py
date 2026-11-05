import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Pad audio/video time axes, preserving the source-to-mouth ordering.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict): padded tensors, lengths, and metadata:
            ids (list[str]): B sample identifiers.
            sample_rates: [B], sample rates in Hz.
            num_samples: [B], original audio lengths from the index.
            audio_lens: [B], original durations in seconds.
            audio_lengths: [B], loaded audio lengths before padding.
            mix_audios: [B, T_max], zero-padded mixtures.
            source_audios (Tensor | None): [B, S, T_max], zero-padded sources,
                or None when ground truth is absent for the entire batch.
            mouth_videos (Tensor | None): [B, S, F_max, H, W], zero-padded
                grayscale frames, or None when video is absent.
            mouth_lengths (Tensor | None): [B, S], frame counts before
                padding, or None when video is absent.
    """

    result_batch = {
        "ids": [item["id"] for item in dataset_items],
        "sample_rates": torch.tensor(
            [item["sample_rate"] for item in dataset_items], dtype=torch.long
        ),
        "num_samples": torch.tensor(
            [item["num_samples"] for item in dataset_items], dtype=torch.long
        ),
        "audio_lens": torch.tensor(
            [item["audio_len"] for item in dataset_items], dtype=torch.float32
        ),
        "audio_lengths": torch.tensor(
            [item["mix_audio"].shape[0] for item in dataset_items], dtype=torch.long
        ),
    }

    batch_size = len(dataset_items)
    num_sources = len(dataset_items[0]["source_audios"])
    for item in dataset_items:
        if (
            not num_sources
            or len(item["source_audios"]) != num_sources
            or len(item["mouth_videos"]) != num_sources
        ):
            raise ValueError(
                "All items must have the same nonzero number of sources and mouths"
            )
        for s_audio in item["source_audios"]:
            if s_audio is not None and s_audio.shape != item["mix_audio"].shape:
                raise ValueError(
                    f"{item['id']}: source audio must match mix_audio shape"
                )

    result_batch["mix_audios"] = pad_sequence(
        [item["mix_audio"] for item in dataset_items], batch_first=True, padding_value=0
    )
    result_batch["mouth_lengths"] = None
    for item_key in ("source_audios", "mouth_videos"):
        values = [value for item in dataset_items for value in item[item_key]]
        if all(value is None for value in values):
            result_batch[item_key] = None
        elif any(value is None for value in values):
            raise ValueError(f"All items must have {item_key} or none of them")
        else:
            padded = pad_sequence(values, batch_first=True, padding_value=0)
            result_batch[item_key] = padded.reshape(
                batch_size, num_sources, *padded.shape[1:]
            )
            if item_key == "mouth_videos":
                result_batch["mouth_lengths"] = torch.tensor(
                    [
                        [mouth.shape[0] for mouth in item["mouth_videos"]]
                        for item in dataset_items
                    ],
                    dtype=torch.long,
                )

    return result_batch
