#!/bin/bash

venv_name=slice-stacker
venv_dir="$HOME/VENV"

echo "Create python venv: $venv_name"
mkdir -p $venv_dir
builtin cd $venv_dir

if [ -d "$venv_name" ]; then
  echo "The $venv_name venv exists. Removing $venv_name now."
  command rm -rf -- $venv_name
fi
echo "Create venv $venv_name for PyMeshLab projects:"
python3 -m venv $venv_name --prompt=$venv_name

source $venv_name/bin/activate
pip install --upgrade pip
pip install --upgrade numpy
pip install --upgrade opencv-python
pip install --upgrade tifffile

deactivate
