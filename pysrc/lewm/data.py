"""Episode-windowed dataset for LeWM training and evaluation.

Matches the windowing rules of ``stable_worldmodel.data.HDF5Dataset``:

- one sample spans ``num_steps * frameskip`` primitive environment steps;
- pixels are read at ``start + k * frameskip`` for ``k = 0 .. num_steps - 1``;
- the action slice ``[start, start + num_steps * frameskip)`` is reshaped to
  ``(num_steps, frameskip * action_dim)``;
- clips are sampled uniformly over every valid start in every episode.

The on-disk layout is the upstream one: flat per-column HDF5 datasets plus
``episode_idx`` (or ``ep_idx``) and ``step_idx`` columns. Episodes must have
contiguous rows and ``step_idx == arange(length)``; this is what the upstream
collector writes and lets clips be read with a single slice.
"""

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_image_transform(img_size=224, resize=True):
    """The reference training transform: uint8 HWC -> normalized float CHW."""
    from torchvision.transforms import v2

    steps = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if resize:
        steps.append(v2.Resize((img_size, img_size), antialias=True))
    return v2.Compose(steps)


@dataclass
class EpisodeInfo:
    first_row: int
    length: int


class EpisodeDataset(Dataset):
    """Clip dataset over a flat HDF5 episode file.

    Args:
        path: HDF5 file with one dataset per column.
        keys: columns to read; must include ``pixels`` and ``action`` for LeWM.
        num_steps: frames per clip (``history_size + num_preds``).
        frameskip: primitive steps between frames.
        transform: callable applied to each raw pixel frame (HWC numpy).
        cache: keep the ``pixels`` and ``action`` columns in memory.
        episode_key/step_key: metadata column names; auto-detected by default.
    """

    def __init__(
        self,
        path,
        keys: Sequence[str] = ("pixels", "action"),
        num_steps: int = 4,
        frameskip: int = 5,
        transform: Callable | None = None,
        cache: bool = False,
        episode_key: str | None = None,
        step_key: str | None = None,
    ):
        import h5py

        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.path = path
        self.keys = tuple(keys)
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.transform = transform
        self.span = num_steps * frameskip

        self._file = h5py.File(path, "r")
        self._cache = {}
        self._episode_key = episode_key or self._detect(
            self._file, ("episode_idx", "ep_idx")
        )
        self._step_key = step_key or "step_idx"
        if self._episode_key is None:
            raise KeyError(
                "no episode column found; expected 'episode_idx' or 'ep_idx'"
            )

        episode_idx = np.asarray(self._file[self._episode_key][:])
        step_idx = np.asarray(self._file[self._step_key][:])
        self._episodes, self.clips = self._build_clips(episode_idx, step_idx)

        if cache:
            for key in self.keys:
                self._cache[key] = np.asarray(self._file[key][:])

    @staticmethod
    def _detect(handle, candidates: Iterable[str]):
        for name in candidates:
            if name in handle:
                return name
        return None

    def _build_clips(self, episode_idx, step_idx):
        order = np.lexsort((step_idx, episode_idx))
        eps = episode_idx[order]
        steps = step_idx[order]
        unique, starts = np.unique(eps, return_index=True)
        starts = np.append(starts, len(order))

        episodes = {}
        clips = []
        for i, ep in enumerate(unique):
            segment = order[starts[i] : starts[i + 1]]
            segment_steps = steps[starts[i] : starts[i + 1]]
            if len(segment) == 0:
                continue
            expected_steps = np.arange(len(segment))
            expected_rows = np.arange(segment[0], segment[0] + len(segment))
            if not (
                np.array_equal(segment_steps, expected_steps)
                and np.array_equal(segment, expected_rows)
            ):
                raise ValueError(
                    f"episode {int(ep)} is not stored as contiguous rows with "
                    "step_idx == arange(length); re-index the file before use"
                )
            episodes[int(ep)] = EpisodeInfo(int(segment[0]), len(segment))

            for start in range(0, len(segment) - self.span + 1):
                clips.append((int(ep), start))

        if not clips:
            raise ValueError(
                f"no clips of {self.span} steps found; longest episode is "
                f"{max((e.length for e in episodes.values()), default=0)} rows"
            )
        return episodes, clips

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.clips)

    def get_dim(self, key):
        return int(self._file[key].shape[-1])

    def episode_length(self, episode):
        return self._episodes[int(episode)].length

    def get_frame(self, episode, step):
        """Transformed pixel frame ``(C, H, W)`` at an absolute episode step."""
        info = self._episodes[int(episode)]
        if not 0 <= step < info.length:
            raise IndexError(f"step {step} outside episode {episode}")
        frame = self._decode_frame(self._read("pixels", info.first_row + step))
        if self.transform is not None:
            return self.transform(frame)
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
        return tensor.float().div(255.0) if tensor.dtype == torch.uint8 else tensor

    def get_action(self, episode, step):
        info = self._episodes[int(episode)]
        return torch.as_tensor(self._read("action", info.first_row + step)).float()

    def _read(self, key, row):
        if key in self._cache:
            return self._cache[key][row]
        return self._file[key][row]

    def _decode_frame(self, frame):
        frame = np.asarray(frame)
        if frame.ndim == 1 and frame.dtype == np.uint8:
            from torchvision.io import decode_jpeg

            frame = decode_jpeg(torch.from_numpy(frame)).permute(1, 2, 0).numpy()
        if frame.ndim != 3:
            raise ValueError(f"expected HWC frame, got shape {frame.shape}")
        return frame

    def __getitem__(self, index):
        ep, start = self.clips[index]
        rows = [start + k * self.frameskip for k in range(self.num_steps)]
        pixels = torch.stack([self.get_frame(ep, step) for step in rows])
        actions = torch.stack(
            [self.get_action(ep, step) for step in range(start, start + self.span)]
        )
        return {"pixels": pixels, "action": actions.reshape(self.num_steps, -1)}

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        # h5py handles are not picklable; workers reopen the file themselves.
        state = self.__dict__.copy()
        state["_file"] = None
        state["_cache"] = {}
        return state

    def __setstate__(self, state):
        import h5py

        self.__dict__.update(state)
        self._file = h5py.File(self.path, "r")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def random_split(dataset, lengths, generator=None):
    """Sample-level random split, matching ``stable_pretraining.data.random_split``."""
    if isinstance(lengths[0], float):
        total = len(dataset)
        counts = [int(np.floor(frac * total)) for frac in lengths]
        remainder = total - sum(counts)
        for i in range(remainder):
            counts[i % len(counts)] += 1
    else:
        counts = list(lengths)
    if sum(counts) != len(dataset):
        raise ValueError(f"split lengths {counts} do not sum to {len(dataset)}")
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    subsets = []
    offset = 0
    for count in counts:
        subsets.append(Subset(dataset, indices[offset : offset + count]))
        offset += count
    return subsets


def split_by_episode(dataset: EpisodeDataset, lengths, generator=None):
    """Episode-level split; keeps windows from one episode in one split."""
    episodes = sorted(dataset._episodes)
    perm = torch.randperm(len(episodes), generator=generator).tolist()
    n = len(episodes)
    if isinstance(lengths[0], float):
        sizes = [int(np.floor(frac * n)) for frac in lengths]
        sizes[0] += n - sum(sizes)
    else:
        sizes = list(lengths)
    if sum(sizes) != n:
        raise ValueError(f"split sizes {sizes} do not sum to {n} episodes")

    parts = []
    offset = 0
    for size in sizes:
        chosen = {episodes[i] for i in perm[offset : offset + size]}
        idx = [i for i, (ep, _) in enumerate(dataset.clips) if ep in chosen]
        parts.append(Subset(dataset, idx))
        offset += size
    return parts
