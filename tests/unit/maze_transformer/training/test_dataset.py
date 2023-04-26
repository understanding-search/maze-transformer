from pathlib import Path
import pytest
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig

class TestGPTDatasetConfig:
    def test_tokenizer_map(self):
        cfg = MazeDatasetConfig(name="test", grid_n=3, n_mazes=1)
        assert list(cfg.tokenizer_map.keys()) == cfg.token_arr
        

class TestGPTDataset:
    class TestFromConfig:
        cfg = MazeDatasetConfig(name="test", grid_n=3, n_mazes=1)
        dataset = MazeDataset.generate(cfg)

        @pytest.mark.usefixtures("temp_dir")
        def test_load_local(self, mocker, temp_dir):
            local_path = temp_dir / Path(f"{self.cfg.to_fname()}.zanj")
            local_path.touch()
            read = mocker.patch.object(MazeDataset, "read")
            download = mocker.patch.object(MazeDataset, "download")
            generate = mocker.patch.object(MazeDataset, "generate")
            read.return_value = self.dataset

            output = MazeDataset.from_config(
                self.cfg,
                local_base_path=temp_dir,
                load_local=True,
                do_download=True,
                do_generate=True,
            )

            assert output == self.dataset
            read.assert_called_once()
            download.assert_not_called()
            generate.assert_not_called()

        @pytest.mark.usefixtures("temp_dir")
        def test_download(self, mocker, temp_dir):
            download = mocker.patch.object(MazeDataset, "download")
            generate = mocker.patch.object(MazeDataset, "generate")
            download.return_value = self.dataset

            output = MazeDataset.from_config(
                self.cfg,
                local_base_path=temp_dir,
                load_local=True,
                do_download=True,
                do_generate=True,
            )

            # We didn't create a local file, so loading should fallback to download
            assert output == self.dataset
            download.assert_called_once()
            generate.assert_not_called()

        @pytest.mark.usefixtures("temp_dir")
        def test_generate(self, mocker, temp_dir):
            # download is not implemented - when it is, we'll need to mock it
            generate = mocker.patch.object(MazeDataset, "generate")
            generate.return_value = self.dataset

            output = MazeDataset.from_config(
                self.cfg,
                local_base_path=temp_dir,
                load_local=True,
                do_download=True,
                do_generate=True,
            )

            assert output == self.dataset
            generate.assert_called_once()

        @pytest.mark.usefixtures("temp_dir")
        def test_all_fail(self, temp_dir):
            with pytest.raises(ValueError):
                MazeDataset.from_config(
                    self.cfg,
                    local_base_path=temp_dir,
                    load_local=True,
                    do_download=True,
                    do_generate=False,
                )

    @pytest.mark.usefixtures("temp_dir")
    def test_save_load(self, temp_dir):
        cfg = MazeDatasetConfig(name="test", grid_n=3, n_mazes=3)
        dataset = MazeDataset.generate(cfg)
        filepath = temp_dir / Path(f"{cfg.to_fname()}.zanj")
        
        dataset.save(filepath)
        loaded = MazeDataset.read(filepath)

        assert dataset == loaded

