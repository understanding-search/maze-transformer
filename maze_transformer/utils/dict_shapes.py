import json
from muutils.dictmagic import dotlist_to_nested_dict

def get_dict_shapes(d: dict[str, "torch.Tensor"]) -> dict[str, tuple[int, ...]]:
	"""given a state dict or cache dict, compute the shapes and put them in a nested dict"""
	return dotlist_to_nested_dict({k: tuple(v.shape) for k, v in d.items()})

def string_dict_shapes(d: dict[str, "torch.Tensor"]) -> str:
	"""printable version of get_dict_shapes"""
	return json.dumps(
		dotlist_to_nested_dict({
			k: str(tuple(v.shape)) # to string, since indent wont play nice with tuples 
			for k, v in d.items()
		}), 
		indent=2,
	)

