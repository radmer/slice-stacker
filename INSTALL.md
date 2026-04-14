# Building and Installing with pyproject.toml

## For development (editable install)

```bash
cd slice-stacker
pip install -e .
```

This installs the package in "editable" mode — changes to your source files take effect immediately without reinstalling. Same as `pip install -e .` with setup.py.

## To build a distributable package

```bash
pip install build
python -m build
```

This creates `dist/slice_stacker-0.1.0.tar.gz` and `dist/slice_stacker-0.1.0-py3-none-any.whl` that you can upload to PyPI or install elsewhere.

## To install from the wheel

```bash
pip install dist/slice_stacker-0.1.0-py3-none-any.whl
```

That's it. The `pyproject.toml` replaces `setup.py`, `setup.cfg`, and `MANIFEST.in` in most cases. The `pip` and `build` tools read it directly.
