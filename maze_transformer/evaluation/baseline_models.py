import random

import numpy as np
import torch
from jaxtyping import Float
from maze_dataset import (
    SPECIAL_TOKENS,
    Coord,
    CoordArray,
    CoordTup,
    LatticeMaze,
    SolvedMaze,
)
from maze_dataset.tokenization.token_utils import (
    get_origin_tokens,
    get_path_tokens,
    get_target_tokens,
)
from maze_dataset.tokenization.util import strings_to_coords
from transformer_lens import HookedTransformer

from maze_transformer.training.config import ConfigHolder


class InvalidTaskForRandomBaselineError(Exception):
    """the task is not coordinate prediction, and is not valid for a random baseline "model" """

    pass


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
        # This conversion won't be needed after https://github.com/understanding-search/maze-transformer/issues/154
        return [tuple(arr.tolist()) for arr in neighbors]

    def _get_all_valid_next_steps(
        self,
        solved_maze: SolvedMaze,
        target: CoordTup,
        path: list[CoordTup] | None = None,
        pad_eos: bool = False,
    ) -> tuple[CoordTup | str | None, list[CoordTup | str]]:
        """returns a tuple of (correct_step, incorrect_steps)"""

        if path is None or len(path) == 0:
            return (tuple(solved_maze.start_pos), [])

        path_end_return: tuple[str, list[str]] = (SPECIAL_TOKENS.PATH_END, [])

        current_position: CoordTup = path[-1]
        # pad with eos up to max_new_tokens to avoid ragged tensors
        if pad_eos:
            if current_position in [target, SPECIAL_TOKENS.PATH_END]:
                return path_end_return
        if current_position == target:
            return path_end_return

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
            return path_end_return
        else:
            if correct_step not in unvisited_neighbors:
                return (None, unvisited_neighbors)

            incorrect_steps = unvisited_neighbors[:]  # what is this doing?
            incorrect_steps.remove(correct_step)

            return (correct_step, incorrect_steps)

    def _predict_next_step(
        self,
        solved_maze: SolvedMaze,
        target: CoordTup,
        path: list[CoordTup],
        pad_eos: bool = False,
    ) -> CoordTup | str:
        """returns a tuple coordinate or a special token"""

        correct_step: CoordTup | str | None
        incorrect_steps: list[CoordTup | str]
        correct_step, incorrect_steps = self._get_all_valid_next_steps(
            solved_maze=solved_maze,
            target=target,
            path=path,
            pad_eos=pad_eos,
        )

        # if only one option, return that
        if len(incorrect_steps) == 0:
            assert correct_step is not None
            return correct_step

        # if no correct choice (no backtracking, towards target), return random choice
        if correct_step is None:
            assert len(incorrect_steps) > 0
            return random.choice(incorrect_steps)

        # if there is a correct choice, choose randomly between correct and incorrect
        n_unvisited_neighbors: int = len(incorrect_steps) + 1
        prob_of_incorrect = (len(incorrect_steps) / n_unvisited_neighbors) * (
            1 - self.bias
        )

        will_choose_correctly = random.random() > prob_of_incorrect
        if will_choose_correctly:
            return correct_step
        else:
            return random.choice(incorrect_steps)

    def _tokens_to_maze_and_path(
        self,
        tokens: list[str],
    ) -> tuple[SolvedMaze, list[Coord]]:
        # assemble the maze from the tokens
        maze: LatticeMaze = LatticeMaze.from_tokens(
            tokens, self.tokenizer._maze_tokenizer
        )
        origin_coord: CoordTup = strings_to_coords(get_origin_tokens(tokens))[0]
        target_coord: CoordTup = strings_to_coords(get_target_tokens(tokens))[0]
        solution: CoordArray = maze.find_shortest_path(origin_coord, target_coord)
        solved_maze: SolvedMaze = SolvedMaze.from_lattice_maze(maze, solution)
        assert (solved_maze.start_pos == np.array(origin_coord)).all()
        assert (solved_maze.end_pos == np.array(target_coord)).all()

        # get the path so far
        context_existing_path: list[Coord] = strings_to_coords(
            get_path_tokens(tokens, trim_end=True),
            when_noncoord="except",
        )

        return (solved_maze, context_existing_path)

    def _generate_path(
        self,
        tokens: list[str],
        steps_to_predict: int,
    ) -> list[str]:
        solved_maze: SolvedMaze
        context_existing_path: list[Coord]
        solved_maze, context_existing_path = self._tokens_to_maze_and_path(tokens)
        origin_coord: CoordTup = tuple(solved_maze.start_pos.tolist())
        target_coord: CoordTup = tuple(solved_maze.end_pos.tolist())

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
            if predictions[-1] == SPECIAL_TOKENS.PATH_END:
                break

        return self.tokenizer._maze_tokenizer.coords_to_strings(
            predictions, when_noncoord="include"
        )

    def _process_context(
        self,
        context: str | list[str] | Float[torch.Tensor, "pos"],
    ) -> list[str]:
        tokens: list[str]
        if isinstance(context, torch.Tensor):
            tokens = self.to_str_tokens(context, prepend_bos=False)
        elif isinstance(context, list):
            if all(isinstance(x, str) for x in context):
                tokens = context
            else:
                raise TypeError(
                    f"Expected list of str, got types in list: {set(type(x) for x in context)}"
                )
        elif isinstance(context, str):
            tokens = self.tokenizer.tokenize(context)
            assert (
                "" not in tokens and " " not in tokens
            ), "Tokenizer error, split `context` includes bad token strings."
        else:
            raise TypeError(f"Expected list[str], str, or tensor, got {type(context)}")

        return tokens

    def generate(
        self,
        context: str | list[str] | Float[torch.Tensor, "pos"],
        max_new_tokens: int,
        **_,
    ) -> str:
        # hack for more than one batch
        if isinstance(context, torch.Tensor):
            if context.ndim == 2:
                return [
                    self.generate(
                        context[i],
                        max_new_tokens,
                    )
                    for i in range(context.shape[0])
                ]

        # convert input to a list of tokens
        tokens: list[str] = self._process_context(context)

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

    def get_valid_next_steps(
        self,
        context: str | list[str] | Float[torch.Tensor, "pos"],
    ) -> tuple[CoordTup | str | None, list[CoordTup | str]]:
        # convert input to a list of tokens
        tokens: list[str] = self._process_context(context)

        # get maze and path
        solved_maze: SolvedMaze
        context_existing_path: list[Coord]
        solved_maze, context_existing_path = self._tokens_to_maze_and_path(tokens)

        # get valid next steps
        return self._get_all_valid_next_steps(
            solved_maze=solved_maze,
            target=tuple(solved_maze.end_pos.tolist()),
            path=context_existing_path,
        )
