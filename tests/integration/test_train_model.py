from pathlib import Path

from maze_dataset.dataset.configs import MAZE_DATASET_CONFIGS

from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.train_model import TrainingResult, train_model
from maze_transformer.training.wandb_logger import WandbProject

temp_dir: Path = Path("tests/_temp/test_train_model")


def test_configs():
    for k, v in MAZE_DATASET_CONFIGS.items():
        assert k == v.to_fname()
        v_summary = v.summary()
        print(f"{k}: {v_summary}")
        assert k == v_summary["fname"]


def test_train_model():
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("demo_small-g3-n100-a_dfs-h44636", "nano-v1", "test-v1"),
    )
    print(cfg.dataset_cfg.summary())
    assert cfg.dataset_cfg.n_mazes == 100
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )

    assert isinstance(result.model, ZanjHookedTransformer)
    assert result.model.zanj_model_config == cfg

    print(cfg.dataset_cfg.summary())


def test_model_loading():
    # get config
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("demo_small-g3-n100-a_dfs-h44636", "nano-v1", "test-v1"),
    )
    print(cfg.dataset_cfg.summary())
    # train model
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )
    # model_ret: ZanjHookedTransformer = result.model

    # # load model
    # zanj: ZANJ = ZANJ(
    #     custom_settings={
    #         "_load_state_dict_wrapper": {"recover_exact": True, "fold_ln": False}
    #     }
    # )
    # model_load_auto: ZanjHookedTransformer = zanj.read(
    #     result.output_path / TRAIN_SAVE_FILES.model_final_zanj
    # )

    # # Load model manually without folding
    # assert cfg == model_ret.zanj_model_config
    # assert_model_cfg_equality(model_ret, model_load_auto)

    # vocab_size: int = len(model_ret.zanj_model_config.tokenizer)
    # assert_model_output_equality(
    #     model_ret,
    #     model_load_auto,
    #     check_argsort_equality=(vocab_size > 2048),
    # )
    # # assert_model_exact_equality(model_ret, model_load_auto)
