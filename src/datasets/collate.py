import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
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

    for item in dataset_items:
        for key in ("s1_audio", "s2_audio"):
            if item[key] is not None and item[key].shape != item["mix_audio"].shape:
                raise ValueError(f"{item['id']}: {key} must match mix_audio shape")

    for key in ("mix_audio", "s1_audio", "s2_audio", "mouth1_video", "mouth2_video"):
        values = [item[key] for item in dataset_items]
        if all(value is None for value in values):
            result_batch[key + "s"] = None
        elif any(value is None for value in values):
            raise ValueError(f"All items must have {key} or none of them")
        else:
            result_batch[key + "s"] = pad_sequence(
                values, batch_first=True, padding_value=0
            )

    for video_key, length_key in (
        ("mouth1_video", "mouth1_lengths"),
        ("mouth2_video", "mouth2_lengths"),
    ):
        result_batch[length_key] = None
        if result_batch[video_key + "s"] is not None:
            result_batch[length_key] = torch.tensor(
                [item[video_key].shape[0] for item in dataset_items], dtype=torch.long
            )

    return result_batch
