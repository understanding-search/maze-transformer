import numpy as np
import IPython
import matplotlib

def colorize(
		tokens: list[str], 
		weights: list[float], 
		cmap: matplotlib.colors.Colormap|str = "Blues",
		template: str = '<span class="barcode"; style="color: black; background-color: {clr}">&nbsp{tok}&nbsp</span>',
	) -> str:
	"""given a sequence of tokens and their weights, colorize the tokens according to the weights (output is html)
	
	originally from https://stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights"""

	if isinstance(cmap, str):
		cmap = matplotlib.cm.get_cmap(cmap)

	colored_string: str = ''

	for word, color in zip(tokens, weights):
		color_hex: str = matplotlib.colors.rgb2hex(cmap(color)[:3])
		colored_string += template.format(clr=color_hex, tok=word)

	return colored_string



def _test():
	mystr: str = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"
	tokens: list[str] = mystr.split()
	weights: list[float] = np.random.rand(len(tokens)).tolist()
	colored: str = colorize(tokens, weights)
	IPython.display.display(IPython.display.HTML(colored))

if __name__ == "__main__":
	_test()