# cfBm-KPZ
Simulating and comparing the point fields associated with coalescing fractional Brownian motion and last passage percolation.

See our paper for more details of the objects being simulated and tested: https://arxiv.org/abs/2112.02247

CfBM_LPP.py contains all documented code generating coalescing fractional Brownian motion and an exactly solvable model of last passage percolation studied in our paper.

KS_test.py contains all documented code that reads our data files and perform the statistical tests.

All data sets use in our paper are included, with file name of the format "[process]_[down/up]_[steps]_[hurst]":
   - [process] specifies which process is being simulated
   - [down/up] indicates whether the data contains the lower or upper (respectively) point fields
   - [steps] indicates the step size used in generating the data, which is 1024 for cfBm models and 4096 for LPP
   - [hurst] (for cfBm only) indicates the hurst index used, which is always 2/3 (written as 2o3, "2 over 3") 

In the data files, each new line is an independent implementation of the point fields, printed directly as a list.
