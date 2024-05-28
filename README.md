# Efficient colour representation for Gaussian Splatting

This repository contains the code for my Master Thesis, "Efficient colour representation for Gaussian Splatting". The main technique introduced is Textured Rendering, where a 3D Gaussian Splatting model is rendered with the training views as textures. By sampling the colours from textures instead of storing them directly in the Gaussian primites, high frequency detail can be modelled with much fewer Gaussian primitives, leading to higher levels of detail in smaller models. For details, see repord.pdf.

## Code Contribution
The code is based on the original 3DGS implementation at https://github.com/graphdeco-inria/gaussian-splatting, where this repository is forked from. The full diff listing all my edits can be found at https://github.com/graphdeco-inria/gaussian-splatting/compare/main...Istlemin:textured-rendering-3dgs:main. The main new files of interest are `textured_render.py` and `depth_images.py`, where the majority of the textured rendering is implemented. In addition to this, substantial changes are made across `render.py`, `train.py` and `diff-gaussian-rasterization/cuda_rasterizer/forward.cu`.  

## Installation
The codebase has the same prerequisites as the original repository, with the only different that the modified `diff-gaussian-rasterization` module needs to be installed.

## Reproducing Results

The dataset used can be found at https://drive.google.com/file/d/1yISt0t4-Eyi1-yJXEXQN0CZvGOd6PBXb/view?usp=sharing. In order to reproduce the main results from the paper, simply run `experiments/evaluate_all.sh` and `experiments/train_texture_all.sh`. Then, running the notebook `notebooks/plot.ipynb` will produce the main plot comparing the performance of my techniques against the baseline. 
