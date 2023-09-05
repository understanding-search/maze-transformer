import functools
import typing

from maze_dataset import SPECIAL_TOKENS


def get_token_first_index(search_token: str, token_list: list[str]) -> int:
    return token_list.index(search_token)


TaskSetup = typing.NamedTuple(
    "TaskSetup",
    [
        ("prompts", list[list[str]]),
        ("targets", str),
    ],
)


class DLAProtocol(typing.Protocol):
    """should take a dataset's tokens, and return a tuple of (prompts, targets)"""

    def __call__(self, dataset_tokens: list[list[str]], **kwargs) -> TaskSetup:
        ...


class DLAProtocolFixed(typing.Protocol):
    """should take a dataset's tokens, and return a tuple of (prompts, targets)

    this variant signifies it's ready to be used -- no keyword arguments are needed
    """

    def __call__(self, dataset_tokens: list[list[str]]) -> TaskSetup:
        ...


def token_after_fixed_start_token(
    dataset_tokens: list[list[str]],
    start_token: str = SPECIAL_TOKENS.PATH_START,
    offset: int = 1,
) -> TaskSetup:
    """in this task, we simply predict the token after `start_token`

    # Parameters:
     - `dataset_tokens : list[list[str]]`
       list of string-lists
     - `start_token : str`
       token to look for
       (defaults to `SPECIAL_TOKENS.PATH_START`)
     - `offset : int`
       which token to predict:
         1: the token after `start_token`, given everything up to and including `start_token`
         0: the token at `start_token`, given everything up to and **not** including `start_token`
       (defaults to `1`)

    # Returns:
     - `TaskSetup`
       tuple of (prompts, targets)
    """

    prompts: list[list[str]] = list()
    targets: list[str] = list()

    for maze_tokens in dataset_tokens:
        path_start_idx: int = get_token_first_index(start_token, maze_tokens)
        prompt_tokens: list[str] = maze_tokens[: path_start_idx + offset]
        prompts.append(prompt_tokens)
        targets.append(maze_tokens[path_start_idx + offset])

    return TaskSetup(prompts=prompts, targets=targets)


LOGIT_ATTRIB_TASKS: dict[str, DLAProtocolFixed] = {
    "path_start": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_START, offset=0
    ),
    "origin_after_path_start": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_START, offset=1
    ),
    "path_end": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_END, offset=0
    ),
    "final_before_path_end": functools.partial(
        token_after_fixed_start_token, start_token=SPECIAL_TOKENS.PATH_END, offset=-1
    ),
}
