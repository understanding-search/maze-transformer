import numpy as np
import matplotlib.pyplot as plt
from muutils.logger import gather_val, gather_stream, get_any_from_stream

def plot_loss(
		log_path: str, 
		convolve_windows: int|list[int]|str = (10, 50, 100),
		raw_loss: bool|str = False,
	):
	"""
	Plot the loss of the model.
	"""
	data_raw: list[tuple] = gather_val(
		file = log_path,
		stream = "train",
		keys = ("iter", "n_sequences", "loss"),
	)

	data_config: list[dict] = gather_stream(
		file = log_path,
		stream = "log_config",
	)

	print(f"{len(data_raw) = }")

	print(data_raw[:20])

	iteration, total_sequences, loss = zip(*data_raw)

	# compute a rolling average
	if isinstance(convolve_windows, int):
		convolve_windows = [convolve_windows]
	elif isinstance(convolve_windows, (list, tuple)):
		pass
	elif isinstance(convolve_windows, str):
		convolve_windows = [int(x) for x in convolve_windows.split(",")]
	else:
		raise ValueError(f"{convolve_windows = }")

	loss_rolling_arr: list[np.ndarray] = [
		np.convolve(loss, np.ones(cv * 2)/(cv * 2), mode="valid")
		for cv in convolve_windows
	]

	if raw_loss:
		raw_loss_fmt: str = raw_loss if isinstance(raw_loss, str) else ","
		plt.plot(total_sequences, loss, raw_loss_fmt, label = "raw losses")
	for cv, loss_rolling in zip(convolve_windows, loss_rolling_arr):
		plt.plot(total_sequences[cv:1-cv], loss_rolling, "-", label = f"rolling avg $(\\pm {cv})$")

	plt.ylabel('Loss')
	plt.xlabel('Total sequences')
	plt.yscale('log')
	title: str = ';  '.join([
		f"dataset={get_any_from_stream(data_config, 'data_cfg')['name']}",
		f"train_config={get_any_from_stream(data_config, 'train_cfg')['name']}",
		f"lr={get_any_from_stream(data_config, 'train_cfg')['optimizer_kwargs']['lr']}",
		f"vocab_size={get_any_from_stream(data_config, 'base_model_cfg')['vocab_size']}",
	])
	plt.title(title)
	plt.legend()
	plt.show()