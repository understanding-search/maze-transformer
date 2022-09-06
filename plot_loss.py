from muutils.logger import gather_val
import numpy as np

def plot_loss(log_path: str, convolve_window: int = 50):
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
	loss_rolling = np.convolve(loss, np.ones(convolve_window * 2)/(convolve_window * 2), mode="valid")

	import matplotlib.pyplot as plt
	plt.plot(total_sequences, loss, '.')
	plt.plot(total_sequences[convolve_window:1-convolve_window], loss_rolling, "-")
	plt.ylabel('Loss')
	plt.xlabel('Total sequences')
	plt.yscale('log')
	plt.show()


if __name__ == "__main__":
	import fire
	fire.Fire(plot_loss)