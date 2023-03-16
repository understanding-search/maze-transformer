import tempfile
from pathlib import Path

import pytest


# When this module becomes unmanageable we can organise the fixtures into multiple modules and import them here
# See https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
@pytest.fixture()
def temp_dir() -> Path:
    data_dir = tempfile.TemporaryDirectory()
    yield Path(data_dir.name)
    data_dir.cleanup()
