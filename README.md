# Spyders: 
Simple PYthon Density Estimator using RuSt


## Description
Uses a fast KDtree ([Kiddo](https://github.com/sdd/kiddo)) to estimate densities using epanechnikov kernels, with individual smoothing lengths. This implementation builds a KDtree from the evaluated points, rather than the underlying density points. 

Python functions for a epanechnikov kde, and an implementation of a Modified Breiman Density Estimator (based on https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.114F/abstract).

The current code works for 2<=n_dim<=7, but it should be simple to extend to higher ndims if needed.

This project was made to fill a niche. The performance is better than other examples I found, entirely due to Kiddo. Room for improvement. Mainly tested for ndim=3.

First Rust project, first public python project. Feedback welcome!

## Install
### Simple (python)
- pip install spyders_density
### Develop (python)
- clone the repository
- pip install .
### Develop (rust)
- clone the repository
- rust, maturin ect...

## Examples
See ./example_notebooks/

## Issues
Kiddo fails to create a tree if there are too many points with the same values (https://github.com/sdd/kiddo/issues/78). No in-built workaround as of yet. In some cases this can be mitigated by adding small numerical noise to data first.
