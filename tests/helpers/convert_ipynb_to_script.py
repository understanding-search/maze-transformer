import json
import argparse

def convert_ipynb(
    notebook: dict, 
    keep_md_cells: bool = True,
    header_comment: str = r'#%%',
) -> str:
    result: list[str] = []
    all_cells: list[dict] = notebook['cells']

    for cell in all_cells:

        cell_type: str = cell['cell_type']

        if keep_md_cells and cell_type == 'markdown':
            result.append(f'{header_comment}\n"""\n{"".join(cell["source"])}\n"""')
        elif cell_type == 'code':
            result.append(f'{header_comment}\n{"".join(cell["source"])}')

    return '\n\n'.join(result)

def main(
    in_file: str,
    out_file: str|None = None,
    strip_md_cells: bool = False,
    header_comment: str = r'#%%',
):
    with open(in_file, 'r') as file:
        notebook: dict = json.load(file)

    converted_script: str = convert_ipynb(notebook, not strip_md_cells, header_comment)

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

    args = parser.parse_args()

    main(args.in_file, args.out_file, args.strip_md_cells, args.header_comment)
