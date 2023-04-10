import random
from typing import Union

import numpy as np
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

from maze_transformer.generation.constants import SPECIAL_TOKENS, Coord, CoordTup
from maze_transformer.generation.lattice_maze import LatticeMaze
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
        self.config = config
        self.bias = bias
        super().__init__(cfg=config.transformer_config, tokenizer=config.tokenizer)

    def _get_coord_neighbors(
        self, maze: LatticeMaze, current_position: CoordTup
    ) -> list[CoordTup]:
        neighbors = maze.get_coord_neighbors(np.array(current_position))
        # This conversion won't be needed after https://github.com/AISC-understanding-search/maze-transformer/issues/154
        return [tuple(arr.tolist()) for arr in neighbors]

    def _predict_next_step(
        self,
        maze: LatticeMaze,
        target: CoordTup,
        path: list[str | CoordTup],
        solution: list[Coord],
    ) -> CoordTup | str:
        current_position = path[-1]
        # pad with eos up to max_new_tokens to avoid ragged tensors
        if current_position in [target, SPECIAL_TOKENS["path_end"]]:
            return SPECIAL_TOKENS["path_end"]

        neighbors = self._get_coord_neighbors(maze, current_position)
        unvisited_neighbors = [coord for coord in neighbors if coord not in path]

        # if the current path is already as long as the solution, there can be no correct next step
        correct_step = solution[len(path)] if len(solution) > len(path) else None

        if len(unvisited_neighbors) == 0:
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
        maze = LatticeMaze.from_tokens(tokens)
        origin_coord = self.config.dataset_cfg.token_node_map[get_origin_token(tokens)]
        target_coord = self.config.dataset_cfg.token_node_map[get_target_token(tokens)]
        solution = maze.find_shortest_path(origin_coord, target_coord)

        existing_path = tokens_to_coords(
            get_path_tokens(tokens), self.config.dataset_cfg
        )

        predictions = []

        for i in range(steps_to_predict):
            path = existing_path + predictions
            predictions.append(
                self._predict_next_step(maze, target_coord, path, solution)
            )

        return coords_to_tokens(
            predictions, self.config.dataset_cfg, when_noncoord="include"
        )

    def generate(
        self,
        indices_batch: Union[str, Float[torch.Tensor, "batch pos"]],
        max_new_tokens: int,
        **_
    ) -> Float[torch.Tensor, "batch pos_plus_new_tokens"]:
        tokens_batch = [self.to_str_tokens(indices) for indices in indices_batch]

        solved_mazes = [
            tokens
            + self._generate_path(
                tokens,
                steps_to_predict=max_new_tokens,
            )
            for tokens in tokens_batch
        ]

        output = self.tokenizer(solved_mazes, is_split_into_words=True)["input_ids"]
        return output
