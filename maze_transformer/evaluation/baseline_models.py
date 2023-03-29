from maze_transformer.generation.latticemaze import LatticeMaze
import random
from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.utils.token_utils import (
    decode_maze_tokens_to_coords,
    get_path_tokens,
    get_target_token,
)
import torch


class RandomBaseline:
    """
    A model that chooses valid paths and never backtracks, but makes random decisions at forks.
    """

    def __init__(self, dataset_config):
        self.config = dataset_config

    def _indices_to_tokens(self, indices):
        return [self.config.token_arr[i] for i in indices]

    def _get_coord_neighbors(self, maze, current_position):
        neighbors = maze.get_coord_neighbors(current_position)
        return [tuple(arr.tolist()) for arr in neighbors]

    def _predict_next_step(self, maze, target, path):
        current_position = path[-1]
        # pad with eos up to max_new_tokens to avoid ragged tensors
        if current_position in [target, SPECIAL_TOKENS["path_end"]]:
            return SPECIAL_TOKENS["path_end"]

        neighbors = self._get_coord_neighbors(maze, current_position)
        unvisited_neighbors = [coord for coord in neighbors if coord not in path]

        if len(unvisited_neighbors) == 0:
            return SPECIAL_TOKENS["path_end"]
        else:
            return random.choice(unvisited_neighbors)

    def _path_to_tokens(self, path):
        tokens = []
        for step in path:
            if step == SPECIAL_TOKENS["path_end"]:
                tokens.append(step)
            else:
                tokens.append(self.config.node_token_map[step])

        return tokens

    def _generate_path(
        self,
        tokens,
        steps_to_predict,
    ):
        maze = LatticeMaze.from_tokens(tokens)
        target_token = get_target_token(tokens)
        target_coord = decode_maze_tokens_to_coords([target_token], self.config)[0]
        existing_path = decode_maze_tokens_to_coords(
            get_path_tokens(tokens), self.config
        )

        predictions = []

        for _ in range(steps_to_predict):
            path = existing_path + predictions
            predictions.append(self._predict_next_step(maze, target_coord, path))

        return self._path_to_tokens(predictions)

    def generate(self, indices_batch, max_new_tokens, **kwargs):
        tokens_batch = [self._indices_to_tokens(indices) for indices in indices_batch]

        preds_batch = [
            self._generate_path(
                tokens,
                steps_to_predict=max_new_tokens,
            )
            for tokens in tokens_batch
        ]

        mazes_with_preds = [
            maze + preds for maze, preds in zip(tokens_batch, preds_batch)
        ]

        output_batch = []
        for tokens in mazes_with_preds:
            output_batch.append(
                torch.tensor(
                    [self.config.tokenizer_map[t] for t in tokens], dtype=torch.long
                )
            )
        return output_batch
