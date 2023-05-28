import random

import numpy as np
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

from maze_transformer.generation.constants import (
    SPECIAL_TOKENS,
    Coord,
    CoordArray,
    CoordTup,
)
from maze_transformer.generation.lattice_maze import LatticeMaze, SolvedMaze
from maze_transformer.training.config import ConfigHolder
from maze_transformer.utils.token_utils import (
    coords_to_tokens,
    get_origin_token,
    get_path_tokens,
    get_target_token,
    tokens_to_coords,
)


class RandomBaseline(HookedTransformer):
    """
    A model that chooses valid paths and never backtracks, but makes random decisions at forks.

    Attributes
    --------
    config: ConfigHolder - Used to create the HookedTransformer

    bias: float - How much to bias the choice of path at forks towards the correct solution. 1 means taking the correct path every time, 0 means choosing randomly
    """

    def __init__(self, config: ConfigHolder, bias: float = 0.0):
        assert isinstance(
            config, ConfigHolder
        ), f"config must be a ConfigHolder, got {str(type(config)) = }"
        self.config: ConfigHolder = config
        self.bias: float = bias
        super().__init__(cfg=config.hooked_transformer_cfg, tokenizer=config.tokenizer)

    def _get_coord_neighbors(
        self, maze: LatticeMaze, current_position: CoordTup
    ) -> list[CoordTup]:
        neighbors = maze.get_coord_neighbors(np.array(current_position, dtype=np.int32))
        # This conversion won't be needed after https://github.com/AISC-understanding-search/maze-transformer/issues/154
        return [tuple(arr.tolist()) for arr in neighbors]

    def _predict_next_step(
        self,
        solved_maze: SolvedMaze,
        target: CoordTup,
        path: list[CoordTup],
        pad_eos: bool = False,
    ) -> CoordTup | str:
        current_position: CoordTup = path[-1]
        # pad with eos up to max_new_tokens to avoid ragged tensors
        if pad_eos:
            if current_position in [target, SPECIAL_TOKENS["path_end"]]:
                return SPECIAL_TOKENS["path_end"]
        if current_position == target:
            return SPECIAL_TOKENS["path_end"]

        neighbors: list[CoordTup] = self._get_coord_neighbors(
            solved_maze, current_position
        )
        unvisited_neighbors: list[CoordTup] = [
            coord for coord in neighbors if coord not in path
        ]

        # if the current path is already as long as the solution, there can be no correct next step
        correct_step: CoordTup = (
            tuple(solved_maze.solution[len(path)])
            if len(solved_maze.solution) > len(path)
            else None
        )

        if len(unvisited_neighbors) == 0:
            # break out if dead end
            return SPECIAL_TOKENS["path_end"]
        else:
            if correct_step not in unvisited_neighbors:
                return random.choice(unvisited_neighbors)

            incorrect_steps = unvisited_neighbors[:]
            incorrect_steps.remove(correct_step)

            prob_of_incorrect = (len(incorrect_steps) / len(unvisited_neighbors)) * (
                1 - self.bias
            )

            will_choose_correctly = random.random() > prob_of_incorrect
            if will_choose_correctly:
                return correct_step
            else:
                return random.choice(incorrect_steps)

    def _generate_path(
        self,
        tokens: list[str],
        steps_to_predict: int,
    ) -> list[str]:
        # assemble the maze from the tokens
        maze: LatticeMaze = LatticeMaze.from_tokens(tokens)
        origin_coord: CoordTup = self.config.dataset_cfg.token_node_map[
            get_origin_token(tokens)
        ]
        target_coord: CoordTup = self.config.dataset_cfg.token_node_map[
            get_target_token(tokens)
        ]
        solution: CoordArray = maze.find_shortest_path(origin_coord, target_coord)
        solved_maze: SolvedMaze = SolvedMaze.from_lattice_maze(maze, solution)
        assert (solved_maze.start_pos == np.array(origin_coord)).all()
        assert (solved_maze.end_pos == np.array(target_coord)).all()

        # get the path so far
        context_existing_path: list[Coord] = tokens_to_coords(
            tokens=get_path_tokens(tokens, trim_end=True),
            maze_data_cfg=self.config.dataset_cfg,
            when_noncoord="except",
        )

        # assemble our predicted path
        predictions: list[Coord] = list()

        if len(context_existing_path) == 0:
            # add the origin to the path if it's not there already
            predictions.append(origin_coord)

        for i in range(steps_to_predict):
            path: list[Coord] = context_existing_path + predictions
            predictions.append(
                self._predict_next_step(
                    solved_maze=solved_maze,
                    target=target_coord,
                    path=path,
                )
            )
            if predictions[-1] == SPECIAL_TOKENS["path_end"]:
                break

        return coords_to_tokens(
            predictions, self.config.dataset_cfg, when_noncoord="include"
        )

    def generate(
        self,
        context: str | list[str] | Float[torch.Tensor, "pos"],
        max_new_tokens: int,
        **_,
    ) -> str:
        # convert input to a list of tokens
        tokens: list[str]
        if isinstance(context, torch.Tensor):
            tokens = self.to_str_tokens(context)
        elif isinstance(context, list):
            if all(isinstance(x, str) for x in tokens):
                tokens = context
            else:
                raise TypeError(
                    f"Expected list of str, got types in list: {set(type(x) for x in context)}"
                )
        elif isinstance(context, str):
            tokens = self.tokenizer.tokenize(context)
        else:
            raise TypeError(f"Expected list[str], str, or tensor, got {type(context)}")

        # generate path
        generated_path: list[str] = self._generate_path(
            tokens,
            steps_to_predict=max_new_tokens,
        )

        # assemble output and return
        output: str = " ".join(tokens + generated_path)

        # print(f"context tokens: {tokens}")
        # print(f"Generated path: {generated_path}")
        # print(f"Output: {output}")

        # output: Float[torch.Tensor, "batch pos_plus_new_tokens"] = self.tokenizer(solved_maze, is_split_into_words=True)["input_ids"]
        return output
