from collections import deque
from functools import cached_property
import inspect
from itertools import chain, product
import random
import sys
import time
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP
from muutils.json_serialize import json_serialize, dataclass_serializer_factory, dataclass_loader_factory, try_catch, JSONitem
# from muutils.defaulterdict import DefaulterDict

