from pathlib import Path

import pytest

from maze_transformer.utils.utils import get_checkpoint_paths_for_run


@pytest.mark.usefixtures("temp_dir")
def test_get_checkpoint_paths_for_run(temp_dir):
    run_path = Path(temp_dir)
    checkpoints_path = run_path / "checkpoints"
    checkpoint1_path = checkpoints_path / "model.iter_123.pt"
    checkpoint2_path = checkpoints_path / "model.iter_456.pt"
    other_path = checkpoints_path / "other_file.txt"

    checkpoints_path.mkdir()
    checkpoint1_path.touch()
    checkpoint2_path.touch()
    other_path.touch()

    checkpoint_paths = get_checkpoint_paths_for_run(run_path)

    assert checkpoint_paths == [(123, checkpoint1_path), (456, checkpoint2_path)]
