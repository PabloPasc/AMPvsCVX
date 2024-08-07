# AMPvsCVX
Experiment comparing AMP vs Convex Programming for Pooled Data with Gaussian design matrix and different noise levels. Also tests initializing AMP algorithm with CVX estimate and true signal B_0.

Required packages: cvxpy, numpy, matplotlib, tikzplotlib

# pool_amp.py
Contains required functions to produce plots.

# iid_cvx_check1.py
Check 1, compares AMP to CVX, produces Fig. 1

# iid_cvx_check2.py
Check 2, 3: compares AMP, CVX to AMP from CVX estimate initialization, AMP from true signal initialization. Produces Fig. 2.

# iid_cvx_check3.py
Check 4: same as previous, but for lower noise levels. Produces Fig. 3.
