---
header-includes: "<style>\nbody {\n  max-width: 50em;\n}\n</style>"
source: https://github.com/knc-neural-calculus/knc-tools
title: todo-inline
updated: '2022-09-09 22:20:09'

metadata:
  files_with_todos: 4
  num_items: 9
  num_unique_tags: 2
  searched_files: 20

cfg:
  config:
    cfg_read: itodo.yml
    searchDir: ../
    file_todo: todo-inline.md
    verbose: false
  read:
    tags:
      list:
      - CRIT
      - TODO
      - FIXME
      - FIX
      - BUG
      - DEBUG
      - UGLY
      - HACK
      - NOTE
      - IDEA
      - REVIEW
      - OPTIMIZE
      - CONFIG
      - '!!!'
      - OLD
    SOURCE_FILES:
    - c
    - cpp
    - h
    - hpp
    - py
    - m
    - tex
    - sh
    - java
    - js
    EXCLUDE:
    - inline_todo.py
    - itodo.yml
    - todo-inline.md
    - itodo.yml
    - todo-inline.md
    - itodo.yml
    - todo-inline.md
    MAX_SEARCH_LEN: 15
    context:
      enabled: true
      lines: 15
  write:
    attr_sort_order:
    - tag
    - file
    - lineNum
    item_format: md_det

# suggested command for conversion to html
cmd: "pandoc todo-inline.md -o todo-inline.html --from markdown+backtick_code_blocks+fenced_code_attributes --standalone --toc --toc-depth 1"
---
# **TODO** -- 6 items
## [`../create_dataset.py`](../create_dataset.py) -- 1 item
 - [ ] TODO: figure out unexpected keyword argument linter error here? 
	(line 55)
	
	<details>

	```python {.numberLines startFrom="55"}
	# TODO: figure out unexpected keyword argument linter error here?
	cfg: MazeDatasetConfig = MazeDatasetConfig(
	    name = name,
	    grid_n = grid_n,
	    n_mazes = n_mazes,
	    **cfg_kwargs,
	)
	# create and solve mazes
	c_start = (0, 0)
	c_end = (cfg.grid_n - 1, cfg.grid_n - 1)
	mazes: list[SolvedMaze] 
	
	with multiprocessing.Pool() as pool:
	```

	</details>

## [`../maze_transformer/training/mazedataset.py`](../maze_transformer/training/mazedataset.py) -- 2 items
 - [ ] TODO: handling of minimum sequence length 
	(line 196)
	
	<details>

	```python {.numberLines startFrom="196"}
	    # TODO: handling of minimum sequence length
	    # last element in mazes_array.idxs whose value is smaller than `idx`
	    sequence_idx: int = torch.searchsorted(self.mazes_array.idxs, idx) - 1
	    # slice the array from the start of the sequence to `idx`, including `idx`
	    end_arr_idx: int = min(
	        idx + 1, # up to end of sequence
	        self.mazes_array.idxs[sequence_idx] + self.cfg.seq_len_max, # up to sequence length cutoff
	    )
	    subseq: ATensor = self.mazes_array.arr[ self.mazes_array.idxs[sequence_idx] : end_arr_idx ]
	    # left-pad the sequence
	    return torch.nn.functional.pad(subseq, (self.cfg.seq_len_max + 1 - len(subseq), 0), value=self.cfg.padding_token_idx)
	def __len__(self) -> int:
	    return len(self.mazes_array.arr)
	```

	</details>

 - [ ] TODO: minimum separation 
	(line 226)
	
	<details>

	```python {.numberLines startFrom="226"}
	# TODO: minimum separation
	# n_min_tgt_dist: int = int(max(maze.grid_shape) * p_min_tgt_dist)
	"""if np.abs(start_node - end_node).sum() < n_min_tgt_dist:
	    # if too close, move end node towards the corner opposite the start node
	    opposite_corner: CoordTup = (
	        maze.grid_shape[0] * round(start_node[0] / maze.grid_shape[0]),
	        maze.grid_shape[1] * round(start_node[1] / maze.grid_shape[1]),
	    )
	    # end_node +=
	"""
	mazes: list[SolvedMaze] = list()
	endpoint_nodes: NDArray[(("maze_idx", cfg.n_mazes), ("start_end", 2), ("coord", 2)), np.int8] = np.random.randint(0, cfg.grid_shape, (cfg.n_mazes, 2, 2))
	```

	</details>

## [`../maze_transformer/training/training.py`](../maze_transformer/training/training.py) -- 3 items
 - [ ] TODO: check (near?) equality between `data_cfg` and `dataset.config` 
	(line 165)
	
	<details>

	```python {.numberLines startFrom="165"}
	# TODO: check (near?) equality between `data_cfg` and `dataset.config` 
	# ensure the override of the sequence length is applied
	# TODO: this is hacky
	dataset.cfg = data_cfg
	logger.log_elapsed_last()
	logger.mem_usage()
	length_stats: StatCounter = StatCounter(dataset.get_all_lengths())
	logger.log({"dataset_seq_len_stats": length_stats.summary()})
	logger.log({"dataset_seq_len_stats": length_stats.serialize()}, lvl=50)
	logger.log(f"loaded {len(dataset)} sequences", 20)
	logger.log("creating dataloader", 10)
	dataloader: DataLoader = DataLoader(
	    dataset, 
	```

	</details>

 - [ ] TODO: this is hacky 
	(line 167)
	
	<details>

	```python {.numberLines startFrom="167"}
	# TODO: this is hacky
	dataset.cfg = data_cfg
	logger.log_elapsed_last()
	logger.mem_usage()
	length_stats: StatCounter = StatCounter(dataset.get_all_lengths())
	logger.log({"dataset_seq_len_stats": length_stats.summary()})
	logger.log({"dataset_seq_len_stats": length_stats.serialize()}, lvl=50)
	logger.log(f"loaded {len(dataset)} sequences", 20)
	logger.log("creating dataloader", 10)
	dataloader: DataLoader = DataLoader(
	    dataset, 
	    batch_size = train_cfg.batch_size,
	    **train_cfg.dataloader_cfg,
	```

	</details>

 - [ ] TODO: this is a bit hacky 
	(line 75)
	
	<details>

	```python {.numberLines startFrom="75"}
	# TODO: this is a bit hacky
	if train_cfg.seq_len_max is not None:
	    data_cfg.seq_len_max = train_cfg.seq_len_max
	# set up paths
	basepath_train: Path = basepath / train_dir
	os.makedirs(basepath_train, exist_ok = True)
	os.makedirs(basepath_train / TRAIN_SAVE_FILES.checkpoints, exist_ok = True)
	with open(basepath_train / TRAIN_SAVE_FILES.cfg, "w") as f:
	    json.dump(json_serialize(data_cfg), f, indent = "\t")
	# set up logger
	logger: Logger = Logger(
	    log_path=Path(basepath_train / TRAIN_SAVE_FILES.log).as_posix(),
	    console_print_threshold=30,
	```

	</details>

# **CONFIG** -- 3 items
## [`../maze_transformer/training/config.py`](../maze_transformer/training/config.py) -- 3 items
 - [ ] CONFIGS_LIST: list[BaseGPTConfig] = [ 
	(line 116)
	
	<details>

	```python {.numberLines startFrom="116"}
	_GPT_CONFIGS_LIST: list[BaseGPTConfig] = [
	    BaseGPTConfig(
	        gpt_cfg_name = "tiny-v1",
	        n_embed=32,
	        n_layer=4,
	        n_head=2,
	    ),
	    BaseGPTConfig(
	        gpt_cfg_name = "medium-v1",
	        n_embed=128,
	        n_layer=8,
	        n_head=4,
	    ),
	]
	```

	</details>

 - [ ] CONFIGS: dict[str, BaseGPTConfig] = { 
	(line 131)
	
	<details>

	```python {.numberLines startFrom="131"}
	GPT_CONFIGS: dict[str, BaseGPTConfig] = {
	    cfg.gpt_cfg_name: cfg 
	    for cfg in _GPT_CONFIGS_LIST
	}
	_TRAINING_CONFIG_LIST: list[TrainConfig] = [
	    TrainConfig(
	        name = "tiny-v1",
	        base_gpt_cfg = GPT_CONFIGS["tiny-v1"],
	        optimizer = torch.optim.RMSprop,
	        optimizer_kwargs = dict(lr = 0.000001),
	        batch_size = 32,
	        dataloader_cfg = dict(
	            shuffle = True,
	```

	</details>

 - [ ] CONFIGS: dict[str, TrainConfig] = { 
	(line 157)
	
	<details>

	```python {.numberLines startFrom="157"}
	TRAINING_CONFIGS: dict[str, TrainConfig] = {
	    cfg.name: cfg 
	    for cfg in _TRAINING_CONFIG_LIST
	}
	```

	</details>


