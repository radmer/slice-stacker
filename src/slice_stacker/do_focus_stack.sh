#!/bin/bash

## Option 1: Max method (hard selection, no blending)
#python focus_stack.py $HOME/working/Stack001/*.tif -o $HOME/working/Stack001_max.tif --method max
#
## Option 2: Pyramid with higher focus power (more aggressive selection)
#python focus_stack.py $HOME/working/Stack001/*.tif -o $HOME/working/Stack001_p4.tif --focus-power 4
#
## Option 3: Even more aggressive
#python focus_stack.py $HOME/working/Stack001/*.tif -o $HOME/working/Stack001_p6.tif --focus-power 6


python focus_stack.py $HOME/working/Stack001/*.tif -o $HOME/working/Stack001_max_edges.tif --method max --focus-measure edges

python focus_stack.py $HOME/working/Stack001/*.tif -o $HOME/working/Stack001_max_edges.tif --method max
