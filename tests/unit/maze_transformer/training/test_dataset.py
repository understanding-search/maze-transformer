from pathlib import Path

import pytest
from maze_dataset import MazeDataset, MazeDatasetConfig

TEMP_DIR: Path = Path("tests/_temp/test_dataset")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class TestGPTDataset:
    class TestFromConfig:
        cfg = MazeDatasetConfig(name="test", grid_n=3, n_mazes=1)
        dataset = MazeDataset.generate(cfg)

        def test_load_local(self, mocker):
            local_path = TEMP_DIR / Path(f"{self.cfg.to_fname()}.zanj")
            local_path.touch()
            read = mocker.patch.object(MazeDataset, "read")
            download = mocker.patch.object(MazeDataset, "download")
            generate = mocker.patch.object(MazeDataset, "generate")
            read.return_value = self.dataset

            output = MazeDataset.from_config(
                self.cfg,
                local_base_path=TEMP_DIR,
                load_local=True,
                do_download=True,
                do_generate=True,
            )

            assert output == self.dataset
            read.assert_called_once()
            download.assert_not_called()
            generate.assert_not_called()

        @pytest.mark.skip
        def test_download(self, mocker):
            download = mocker.patch.object(MazeDataset, "download")
            generate = mocker.patch.object(MazeDataset, "generate")
            download.return_value = self.dataset

            output = MazeDataset.from_config(
                self.cfg,
                local_base_path=TEMP_DIR,
                load_local=False,
                do_download=True,
                do_generate=False,
            )

            # We didn't create a local file, so loading should fallback to download
            assert output == self.dataset
            download.assert_called_once()
            generate.assert_not_called()

        def test_generate(self, mocker):
            # download is not implemented - when it is, we'll need to mock it
            generate = mocker.patch.object(MazeDataset, "generate")
            generate.return_value = self.dataset

            output = MazeDataset.from_config(
                self.cfg,
                local_base_path=TEMP_DIR,
                load_local=False,
                do_download=False,
                do_generate=True,
            )

            assert output == self.dataset
            generate.assert_called_once()

        def test_all_fail(self):
            with pytest.raises(ValueError):
                MazeDataset.from_config(
                    self.cfg,
                    local_base_path=TEMP_DIR,
                    load_local=False,
                    do_download=False,
                    do_generate=False,
                )

    def test_save_load(self):
        cfg = MazeDatasetConfig(name="test", grid_n=3, n_mazes=3)
        dataset = MazeDataset.generate(cfg)
        filepath = TEMP_DIR / Path(f"{cfg.to_fname()}.zanj")

        dataset.save(filepath)
        loaded = MazeDataset.read(filepath)

        assert dataset.cfg.diff(loaded.cfg) == {}
        assert dataset.cfg == loaded.cfg
        for x, y in zip(dataset, loaded):
            assert x.diff(y) == {}
