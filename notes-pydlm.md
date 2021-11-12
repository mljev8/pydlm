## To-Do
- Simple variance matrix estimation (non-parametric, form-free)
- Likelihood calculations
- Catalog of known/treated examples (convergence, likelihood values, theoretical MSE)
- Write unit tests
- Write setup.py file (conda install)

## Wishlist
- Support for discount models
- Support for pseudo-observations
- Pre-iterate constant models to steady-state
- DLM_A + DLM_B (block diag G and W, stack F, sum V), ever useful in practice?
- Support for the most advanced and time dynamic case where all components change every single iteration
- Update state model (G,W) separately, update obs model (F,V)
- Treat missing observations (fixed update interval), i.e. handle dropout
- Dynamic GLM's (Poisson, Logistic, ...)

## Investigations
- Particle model, forced physical consistency, zero out entries in W
- Particle model, sampling rate robustness, double up and cut in half
- Multivariate time-series data with strongly coupled components (cantilever beam)
- Block-type covariance structures (2D, 3D, Markov, ...)
- Theoretical RMSE, steady-state performance

## Examples
- BBak
- Bluetooth tracking, incl. smoothing
- Image processing (CD)
- Cantilever beam, coupled TS data (performance with and without coupling)
- Particle motion, physically consistent filtering, parameter estimation
- ADSB smear quantization (offline, smoother)
