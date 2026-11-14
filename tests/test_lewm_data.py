"""Dataset windowing tests on a synthetic HDF5 file (no real dataset needed)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

from lewm.data import EpisodeDataset, random_split, split_by_episode  # noqa: E402

h5py = pytest.importorskip("h5py")


def _write_episodes(path, lengths=(10, 12), action_dim=2, shape=(2, 2, 3)):
    total = sum(lengths)
    pixels = np.zeros((total, *shape), dtype=np.uint8)
    actions = np.zeros((total, action_dim), dtype=np.float32)
    episode_idx = np.zeros(total, dtype=np.int64)
    step_idx = np.zeros(total, dtype=np.int64)
    offset = 0
    for ep, length in enumerate(lengths):
        for step in range(length):
            row = offset + step
            pixels[row] = row
            actions[row] = row
            episode_idx[row] = ep
            step_idx[row] = step
        offset += length
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("episode_idx", data=episode_idx)
        handle.create_dataset("step_idx", data=step_idx)


def test_clip_windows_and_actions(tmp_path):
    path = tmp_path / "episodes.h5"
    _write_episodes(path)

    ds = EpisodeDataset(path, num_steps=3, frameskip=2)
    # lengths 10 and 12, span 6 -> (10-6+1) + (12-6+1) = 12 clips
    assert len(ds) == 12
    assert ds.get_dim("action") == 2

    first = ds[0]
    assert first["pixels"].shape == (3, 3, 2, 2)
    assert first["pixels"].dtype == torch.float32
    # uint8 constant row index / 255
    assert torch.allclose(first["pixels"][:, 0, 0, 0], torch.tensor([0, 2, 4]) / 255.0)
    # actions rows 0..5 reshaped to (num_steps, frameskip * action_dim); each
    # synthetic row stores (row, row)
    expected = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=torch.float32)
    torch.testing.assert_close(first["action"], expected)

    last = ds[-1]
    # episode 2 starts at row 10; its last start is 6 -> rows 16, 18, 20
    assert torch.allclose(last["pixels"][:, 0, 0, 0], torch.tensor([16, 18, 20]) / 255.0)
    torch.testing.assert_close(
        last["action"],
        torch.tensor(
            [[16, 16, 17, 17], [18, 18, 19, 19], [20, 20, 21, 21]],
            dtype=torch.float32,
        ),
    )


def test_transform_is_applied_per_frame(tmp_path):
    path = tmp_path / "episodes.h5"
    _write_episodes(path)

    calls = []

    def transform(frame):
        calls.append(frame.shape)
        return torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)

    ds = EpisodeDataset(path, num_steps=2, frameskip=1, transform=transform)
    item = ds[0]
    assert item["pixels"].shape == (2, 3, 2, 2)
    assert calls == [(2, 2, 3), (2, 2, 3)]


def test_rejects_non_contiguous_episodes(tmp_path):
    path = tmp_path / "bad.h5"
    _write_episodes(path)
    with h5py.File(path, "r") as src:
        pixels = src["pixels"][:]
        actions = src["action"][:]
        episode_idx = src["episode_idx"][:]
        step_idx = src["step_idx"][:]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("episode_idx", data=episode_idx)
        handle.create_dataset("step_idx", data=step_idx + (episode_idx == 1) * 3)

    with pytest.raises(ValueError, match="contiguous"):
        EpisodeDataset(path, num_steps=3, frameskip=2)


def test_random_split_matches_sample_level_semantics(tmp_path):
    path = tmp_path / "episodes.h5"
    _write_episodes(path)
    ds = EpisodeDataset(path, num_steps=3, frameskip=2)

    splits = random_split(ds, [0.75, 0.25], generator=torch.Generator().manual_seed(0))
    assert [len(s) for s in splits] == [9, 3]
    assert sorted(i for s in splits for i in s.indices) == list(range(len(ds)))

    again = random_split(
        ds, [0.75, 0.25], generator=torch.Generator().manual_seed(0)
    )
    assert [s.indices for s in splits] == [s.indices for s in again]


def test_split_by_episode_keeps_episodes_whole(tmp_path):
    path = tmp_path / "episodes.h5"
    _write_episodes(path, lengths=(6, 6, 6, 6))
    ds = EpisodeDataset(path, num_steps=2, frameskip=2)

    train, val = split_by_episode(
        ds, [0.5, 0.5], generator=torch.Generator().manual_seed(0)
    )
    train_eps = {ds.clips[i][0] for i in train.indices}
    val_eps = {ds.clips[i][0] for i in val.indices}
    assert not (train_eps & val_eps)
    assert train_eps | val_eps == {0, 1, 2, 3}
    assert len(train) + len(val) == len(ds)
