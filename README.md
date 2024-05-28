# Efficient colour representation for Gaussian Splatting

This repository contains the code for my Master Thesis, "Efficient colour representation for Gaussian Splatting". The main technique introduced is Textured Rendering, where a 3D Gaussian Splatting model is rendered with the training views as textures. By sampling the colours from textures instead of storing them directly in the Gaussian primites, high frequency detail can be modelled with much fewer Gaussian primitives, leading to higher levels of detail in smaller models. For details, see repord.pdf.

## Code Contribution
The code is based on the original 3DGS implementation at https://github.com/graphdeco-inria/gaussian-splatting, where this repository is forked from. The main 

## Installation
The codebase has the same prerequisites as the 

## Reproducing Results

In order to reproduce the main results from the 
