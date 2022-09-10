set -x

cd docs/

python -m inline_todo.run
pandoc todo-inline.md -o todo-inline.html --from markdown+backtick_code_blocks+fenced_code_attributes --standalone --toc --toc-depth 1