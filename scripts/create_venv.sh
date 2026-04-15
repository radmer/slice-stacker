#!/bin/bash

venv_name=omsystems
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
pip install --upgrade scipy
pip install --upgrade exifread

#pip install --upgrade pillow
#pip install --upgrade prequests
#pip install --upgrade olympuswifi

#pip install --upgrade matplotlib
#pip install --upgrade scipy
#pip install --upgrade nibabel
#pip install --upgrade scikit-image
#pip install --upgrade numpy-stl
#pip install --upgrade trimesh
#pip install --upgrade "napari[all]"
#
#pip install --upgrade pyvista
#pip install --upgrade sympy
#pip install --upgrade pymeshlab
#pip install --upgrade polyscope
deactivate
