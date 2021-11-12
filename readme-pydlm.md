# PyDLM

Kalman filtering. Kalman smoothing. Parameter estimation. 

Abstracted classes prepared for inheritance and application-tailoring.
Various standard models provided in specialized form, ready for use "as is".
Example folder showcases usage, both on simulated and real datasets.

## Design comments
The abstract base class (`DLM_abc`) serves merely as documentation. In principle, this construct 
is "valid" as soon as the member functions `.init_dlm()` and `.forward_filter()` are implemented.
It is not necessary to formalize/implement a Kalman filter as usually suggested in academic textbooks.
The default recursion formulas are oftentimes helpful, especially for comprehension, but they're not mandatory.
A stub version of `.backward_smooth()` will make do if smoothing functionality is not needed. 
I'm striving to maximize a combination of flexibility and code reuseability. Want all models to originate from the same general interface.

The self-contained class `DLM_default_body` is worth a careful study. You will most often inherit from 
this class and make the adjustments necessary for your specific problem at hand. Override as needed, that is.
Most of the specialized constructs/tools in `DLM_tools.py` inherit directly from `DLM_default_body` and make as many (or few) adjustments as needed.

## Examples
bbak (local level + single harmonic)  

## References
[PPC] G. Petris, S. Petrone & P. Campagnoli, Dynamic Linear Models with R, Springer, 2009  
[CD] C. Dethlefsen, Space Time Problems and Applications, Aalborg University, 2002  
[EVS] E. V. Stansfield, Introduction To Kalman Filters, IEE Signal Processing, 2001  
[W&H] M. West & J. Harrison, Bayesian Forecasting and Dynamic Models, 2nd Edition, Springer, 1999  
