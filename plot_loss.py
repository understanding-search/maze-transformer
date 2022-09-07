from muutils.logger import gather_val
import numpy as np

def plot_loss(
		log_path: str, 
		convolve_windows: int|list[int]|str = [10, 50, 100],
		raw_loss: bool = False,
	):
	"""
	Plot the loss of the model.
	"""
	data_raw: list[tuple] = gather_val(
		file = log_path,
		stream = "train",
		keys = ("iter", "n_sequences", "loss"),
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

	import matplotlib.pyplot as plt

	if raw_loss:
		plt.plot(total_sequences, loss, ',', label = "raw losses")
	for cv, loss_rolling in zip(convolve_windows, loss_rolling_arr):
		plt.plot(total_sequences[cv:1-cv], loss_rolling, "-", label = f"rolling avg $(\\pm {cv})$")

	plt.ylabel('Loss')
	plt.xlabel('Total sequences')
	plt.yscale('log')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	import fire
	fire.Fire(plot_loss)