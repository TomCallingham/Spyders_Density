# Spyders: 
Simple PYthon Density Estimator using RuSt

First Rust project, first public python project. Feedback welcome!

## Description
Uses a fast KDtree (Kiddo) to estimate densities using epanechnikov kernels. This implementation builds a KDtree from the evaluated points, rather than the underlying density points. This allows each underlying density point to have a different smoothing length.

## Install
### Simple (python)
- pip install spyder_density
### Develop (python)
- clone the repository
- pip install .
### Develop (rust)
- clone the repository
- rust, maturin ect...

## Issues
Kiddo fails to create a tree if there are too many points with the same values https://github.com/sdd/kiddo/issues/78
