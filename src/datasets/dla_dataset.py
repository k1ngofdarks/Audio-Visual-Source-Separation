import numpy as np
import soundfile as sf
import json
from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DLADataset(BaseDataset):
    """Index the DLA split layout and reuse shared audio/video loading."""

    def __init__(self, split, data_dir=None, index_dir=None, audio_only=False, force_rebuild_index=False, required_sources=None, *args, **kwargs):
        """Prepare a split index and initialize the shared dataset.

        Args:
            split (str): partition name, such as train, val, or test.
            data_dir (str | Path | None): root containing audio/ and mouths/.
            index_dir (str | Path | None): JSON cache directory, writable on rebuild.
            audio_only (bool): skip loading mouth crops when True.
            force_rebuild_index (bool): rebuild even if an index is cached.
            required_sources (bool | None): require both sources; None defaults
                to True for train/val and False for other splits.
            **kwargs: BaseDataset options, including limit and transforms.
        """

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "dla_dataset"

        self._data_dir = Path(data_dir)

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "indexes" / "dla_dataset"

        self._index_dir = Path(index_dir)
        self._split = split
        self._audio_only = audio_only
        self._required_sources = split in ("train", "val") if required_sources is None else required_sources
        
        index = self._get_or_prepare_index(force_rebuild_index=force_rebuild_index)
        super().__init__(index, data_dir=self._data_dir, audio_only=self._audio_only, required_sources=self._required_sources, *args, **kwargs)

    def _get_or_prepare_index(self, force_rebuild_index: bool):
        """Read the cache or build, validate, and save a full split index.

        Returns:
            list[dict]: metadata before applying shuffle or limit.
        """
        index_path = self._index_dir / f"{self._split}_index.json"
        if index_path.exists() and not force_rebuild_index:
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            self._assert_index_is_valid(index)
            self._index_dir.mkdir(parents=True, exist_ok=True)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index


    def _create_index(self):
        """Match mixtures to sources and mouth crops using names and WAV headers.

        Returns:
            list[dict]: relative paths and metadata, with None for missing
                sources. audio_len is in seconds; num_samples is per channel.
        """
        index = []
    
        mix_dir = self._data_dir / "audio" / self._split / "mix"
        for mix_path in sorted(mix_dir.glob("*.wav")):
            s1_path = self._data_dir / "audio" / self._split / "s1" / mix_path.name
            s2_path = self._data_dir / "audio" / self._split / "s2" / mix_path.name
            s1_path = s1_path if s1_path.is_file() else None
            s2_path = s2_path if s2_path.is_file() else None

            first_id, second_id = mix_path.stem.split("_")
            mouth1_path = self._data_dir / "mouths" / f"{first_id}.npz"
            mouth2_path = self._data_dir / "mouths" / f"{second_id}.npz"

            audio_info = sf.info(mix_path)

            index.append({
                "id": mix_path.stem,
                "mix_path": mix_path.relative_to(self._data_dir).as_posix(),
                "s1_path": s1_path.relative_to(self._data_dir).as_posix() if s1_path is not None else None,
                "s2_path": s2_path.relative_to(self._data_dir).as_posix() if s2_path is not None else None,
                "mouth1_path": mouth1_path.relative_to(self._data_dir).as_posix(),
                "mouth2_path": mouth2_path.relative_to(self._data_dir).as_posix(),
                "sample_rate": audio_info.samplerate,
                "num_samples": audio_info.frames,
                "audio_len": audio_info.duration
            })

        return index
