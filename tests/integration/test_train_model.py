from pathlib import Path

from zanj import ZANJ
from zanj.torchutil import assert_model_cfg_equality

from maze_transformer.test_helpers.assertions import assert_model_output_equality
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.train_model import TrainingResult, train_model
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.wandb_logger import WandbProject

temp_dir: Path = Path("tests/_temp/test_train_model")


def test_train_model():
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs-h73257", "nano-v1", "test-v1"),
    )
    cfg.dataset_cfg.n_mazes = 10
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )

    assert isinstance(result.model, ZanjHookedTransformer)
    assert result.model.zanj_model_config == cfg


def test_model_loading():
    zanj: ZANJ = ZANJ(
        custom_settings={
            "_load_state_dict_wrapper": {"recover_exact": True, "fold_ln": False}
        }
    )
    # get config
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs-h73257", "nano-v1", "test-v1"),
    )
    # train model
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )
    model_ret: ZanjHookedTransformer = result.model

    # load model
    model_load_auto: ZanjHookedTransformer = zanj.read(
        result.output_path / TRAIN_SAVE_FILES.model_final_zanj
    )

    # Load model manually without folding
    assert cfg == model_ret.zanj_model_config
    assert_model_cfg_equality(model_ret, model_load_auto)

    vocab_size: int = len(model_ret.zanj_model_config.tokenizer)
    assert_model_output_equality(
        model_ret,
        model_load_auto,
        check_argsort_equality=(vocab_size > 2048),
    )
    # assert_model_exact_equality(model_ret, model_load_auto)
