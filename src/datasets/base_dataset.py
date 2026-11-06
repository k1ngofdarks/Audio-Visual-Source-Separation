import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        data_dir,
        audio_only=False,
        required_sources=True,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): mixture, source, and mouth-crop paths with metadata.
            data_dir (str | Path): root for relative paths stored in the index.
            audio_only (bool): skip video loading and video-file checks.
            required_sources (bool): reject samples without both source audios.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._data_dir = Path(data_dir)
        self._audio_only = audio_only
        self._required_sources = required_sources

        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): audio tensors (mono: [T]), mouth crops
                [frames, H, W], ID, and original audio metadata. Missing sources
                and skipped videos are None. Instance transforms run last.
        """
        data_dict = self._index[ind]
        id = data_dict["id"]

        mix_path = self._data_dir / data_dict["mix_path"]
        source_paths = [
            self._data_dir / path if path is not None else None
            for path in data_dict["source_paths"]
        ]
        mouth_paths = [
            self._data_dir / path if path is not None else None
            for path in data_dict["mouth_paths"]
        ]

        sample_rate = data_dict["sample_rate"]
        num_samples = data_dict["num_samples"]
        audio_len = data_dict["audio_len"]

        instance_data = {
            "id": id,
            "mix_audio": self.load_audio(mix_path),
            "source_audios": [self.load_audio(s_path) for s_path in source_paths],
            "mouth_videos": [
                self.load_video(m_path) if not self._audio_only else None
                for m_path in mouth_paths
            ],
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "audio_len": audio_len,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, audio_path: Path | None):
        """
        Load audio from the given path.

        Args:
            audio_path (Path | None): full audio path, or None for an absent source.
        Returns:
            Tensor | None: float32 waveform (mono: [T]), or None.
        """
        if audio_path is None:
            return None
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio)
        return audio

    def load_video(self, video_path: Path | None):
        """
        Load video from the given path.

        Args:
            video_path (Path | None): full NPZ path, or None for absent video.
        Returns:
            Tensor | None: frames from the 'data' array, shaped [T, H, W],
                with the stored dtype and no normalization, or None.
        """
        if video_path is None:
            return None
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with np.load(video_path, allow_pickle=False) as archive:
            video = torch.from_numpy(archive["data"])

        return video

    def _apply_transform(self, transform, value):
        if value is None:
            return None
        if isinstance(value, list):
            return [self._apply_transform(transform, item) for item in value]
        return transform(value)

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self._apply_transform(
                    self.instance_transforms[transform_name],
                    instance_data[transform_name],
                )
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Optional filtering hook; currently unimplemented and unused.

        If needed, return records passing the chosen metadata criteria and call
        this hook before shuffling and limiting the index.
        """
        # Filter logic
        pass

    def _assert_index_is_valid(self, index):
        """
        Check required fields, file availability, and complete source pairs.

        Sources are mandatory when required_sources=True; video files are
        checked only when audio_only=False. Does not load audio or video.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information.
        """

        assert index, f"No samples found in {self._data_dir}"

        required_fields = (
            "id",
            "mix_path",
            "source_paths",
            "mouth_paths",
            "sample_rate",
            "num_samples",
            "audio_len",
        )
        for entry in index:
            for field in required_fields:
                assert (
                    field in entry
                ), f"Each dataset item should include field '{field}'"

            sample_id = entry["id"]
            assert entry["mix_path"], f"{sample_id}: mix_path must be provided"
            assert (
                self._data_dir / entry["mix_path"]
            ).is_file(), f"{sample_id}: mix file does not exist"

            s_paths = entry["source_paths"]
            m_paths = entry["mouth_paths"]

            assert isinstance(s_paths, list) and isinstance(m_paths, list)
            assert len(s_paths) > 0 and len(s_paths) == len(
                m_paths
            ), f"{sample_id}: sources and mouths must have equal nonzero lengths"

            if not (all(s_paths) == any(s_paths)):
                assert False, "All source paths must be all None or all provided"

            for s_path in s_paths:
                if s_path is not None:
                    assert (
                        self._data_dir / s_path
                    ).is_file(), f"{sample_id}: source file does not exist"

            if self._required_sources:
                assert all(s_paths), "All sources are required"

            if not self._audio_only:
                for mouth_path in m_paths:
                    assert (
                        mouth_path and (self._data_dir / mouth_path).is_file()
                    ), f"{sample_id}: mouth file does not exist"

    @staticmethod
    def _sort_index(index):
        """
        Optional sorting template; currently unused.

        Replace KEY_FOR_SORTING with a real metadata field before calling it.
        Returns a sorted copy; apply before limiting if order matters.
        """
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
