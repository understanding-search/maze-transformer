import json
import argparse
import typing

def convert_ipynb(
    notebook: dict, 
    keep_md_cells: bool = True,
    header_comment: str = r'#%%',
    disable_plots: bool = False,
    filter_out_lines: str|typing.Sequence[str] = ('%', '!'), # ignore notebook magic commands and shell commands
) -> str:
    """Convert Jupyter Notebook to a script, doing some basic filtering and formatting.

    # Arguments
        - `notebook: dict`: Jupyter Notebook loaded as json.
        - `keep_md_cells: bool = True`: Keep markdown cells in the output script.
        - `header_comment: str = r'#%%'`: Comment string to separate cells in the output script.
        - `disable_plots: bool = False`: Disable plots in the output script.
        - `filter_out_lines: str|typing.Sequence[str] = ('%', '!')`: comment out lines starting with these strings (in code blocks). 
            if a string is passed, it will be split by char and each char will be treated as a separate filter.

    # Returns
        - `str`: Converted script.
    """

    if isinstance(filter_out_lines, str):
        filter_out_lines = tuple(filter_out_lines)
    filter_out_lines_set: set = set(filter_out_lines)

    result: list[str] = []
    
    if disable_plots:
        result.extend([
            'import matplotlib.pyplot as plt',
            'plt.ioff()',
        ])

    all_cells: list[dict] = notebook['cells']

    for cell in all_cells:

        cell_type: str = cell['cell_type']

        if keep_md_cells and cell_type == 'markdown':
            result.append(f'{header_comment}\n"""\n{"".join(cell["source"])}\n"""')
        elif cell_type == 'code':
            source: list[str] = cell['source']
            if filter_out_lines:
                source = [
                    f'#{line}' 
                    if line.startswith(filter_out_lines_set) 
                    else line 
                    for line in source
                ]
            result.append(f'{header_comment}\n{"".join(source)}')
            
    return '\n\n'.join(result)

def main(
    in_file: str,
    out_file: str|None = None,
    strip_md_cells: bool = False,
    header_comment: str = r'#%%',
    disable_plots: bool = False,
    filter_out_lines: str|typing.Sequence[str] = ('%', '!'),
):
    with open(in_file, 'r') as file:
        notebook: dict = json.load(file)

    converted_script: str = convert_ipynb(
        notebook=notebook,
        keep_md_cells=not strip_md_cells,
        header_comment=header_comment,
        disable_plots=disable_plots,
        filter_out_lines=filter_out_lines,
    )

    if out_file:
        with open(out_file, 'w') as file:
            file.write(converted_script)
    else:
        print(converted_script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Jupyter Notebook to a script with cell separators.')
    parser.add_argument('in_file', type=str, help='Input Jupyter Notebook file (.ipynb).')
    parser.add_argument('-o', '--out_file', type=str, help='Output script file. If not specified, the result will be printed to stdout.')
    parser.add_argument('--strip_md_cells', action='store_true', help='Remove markdown cells from the output script.')
    parser.add_argument('--header_comment', type=str, default=r'#%%', help='Comment string to separate cells in the output script.')
    parser.add_argument('--disable_plots', action='store_true', help='Disable plots in the output script. Useful for testing in CI.')
    parser.add_argument('--filter_out_lines', type=str, default='%', help='Comment out lines starting with these characters.')

    args = parser.parse_args()

    main(
        in_file=args.in_file,
        out_file=args.out_file,
        strip_md_cells=args.strip_md_cells,
        header_comment=args.header_comment,
        disable_plots=args.disable_plots,
        filter_out_lines=args.filter_out_lines,
    )
