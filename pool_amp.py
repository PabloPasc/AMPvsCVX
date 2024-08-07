#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:47:42 2023

@author: pp423
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog
import cvxpy as cp

# Quantize entries of beta to 0, 1 according to threshold
def quantize_new(beta_hat, thresh):
    beta_hat[beta_hat < thresh] = 0
    beta_hat[beta_hat >= thresh] = 1
    return beta_hat


''' Alternative Optimization methods - Linear Programming and Convex'''

def run_LP(n, p, L, Y, X, B_prop):

  # Configuring our inputs to suit LP
  Y_LP = Y.flatten('F')
  X_LP = np.zeros((n*L,p*L))
  for l in range(L):
    X_LP[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_LP = np.eye(p)
  for l in range(L-1):
    C_LP = np.concatenate((C_LP, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective vector
  c = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    c -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = np.concatenate((X_LP, C_LP), axis=0)
  # Setting the equality constraints vector
  b = np.concatenate((Y_LP, one_p), axis=0)

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Solving linear programming problem
  res = linprog(c, A_eq=A, b_eq=b, A_ub=G, b_ub=h)
  print("Linear program:", res.message)

  # Reconfiguring our outputs to suit pooled data
  B_LP_est = res.x
  print(B_LP_est)
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_LP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est

def run_NP(n, p, L, Y, X, sigma, B_prop):

  # Configuring our inputs to suit CVX
  Y_opt = Y.flatten('F')
  X_opt = np.zeros((n*L,p*L))
  for l in range(L):
    X_opt[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_opt = np.eye(p)
  for l in range(L-1):
    C_opt = np.concatenate((C_opt, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective matrix and vector
  q = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    q -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = C_opt
  # Setting the equality constraints vector
  b = one_p

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Define and solve the CVXPY problem
  constant = 1/(2*p*(sigma**2))
  B_opt = cp.Variable(p*L)
  objective = cp.Minimize(constant*cp.sum_squares(Y_opt - X_opt @ B_opt) + (q.T @ B_opt))
  constraints = []
  constraints.append(G @ B_opt <= h)
  constraints.append(A @ B_opt == b)
  problem = cp.Problem(objective, constraints)
  problem.solve(solver=cp.OSQP)
  print('optimal obj value:', problem.value)

  # Reconfiguring our outputs to suit pooled data
  B_QP_est = B_opt.value
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_QP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est


'''=== Create iid design matrix ==='''
def create_X_iid(alpha, m, n):
    X = np.random.binomial(1, alpha, (m, n))
    return X

'''=== spatially coupled design matrix ==='''
def create_base_matrix(lam, omega, alpha):
  R = lam + omega - 1
  C = lam
  W = np.zeros((R, C))
  for c in range(C):
      for r in range(c, c + omega):
        if alpha >= 0.5:
          W[r, c] = (1 + np.sqrt(1-4*alpha*(1-alpha)/omega)) / (2*alpha)
        elif alpha < 0.5:
          W[r, c] = (1 - np.sqrt(1-4*alpha*(1-alpha)/omega)) / (2*alpha)
  return W

def create_SC_matrix(W, N, alpha, delin):
  R, C = np.shape(W)
  M = int(delin*N)
  m, n = M*R, N*C
  print(m,n)
  X_sc = np.zeros((m,n))
  for c in range(C):
      for r in range(R):
          X_sc[M*r:(M*r+M), N*c:(N*c + N )] = np.random.binomial(1, alpha * W[r,c], (M, N))
  return X_sc

'''=== Create signal matrix B with probabilities vector pi ==='''
def create_B(pi, n):
    assert np.sum(pi)==1
    L = len(pi)
    B_int = np.random.choice(L,n,p=pi)
    B = np.zeros((n, L))
    for i in range(len(B)):
        B[i, B_int[i]] = 1
    return B

'''=== Transform iid design matrix for AMP ==='''
def X_iid_to_X_tilde(X_iid, alpha):
    m, n = np.shape(X_iid)
    X_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(X_iid - alpha)
    return X_tilde


def Xsc_to_Xsctilde(X_sc, W, alpha):
    m, n = np.shape(X_sc)
    R, C = np.shape(W)
    M, N = int(m/R), int(n/C)
    lam = C
    omega = R + 1 - lam
    X_sc_tilde = np.zeros((m,n))
    for c in range(C):
        for r in range(c, c + omega):
            X_sc_tilde[M*r:(M*r+M), N*c:(N*c + N )] = (1/np.sqrt(m*alpha*(1-alpha)/R))*(X_sc[M*r:(M*r+M), N*c:(N*c + N )] - alpha*W[r,c])
    return X_sc_tilde        

'''=== Transform measurements matrix for AMP ==='''            
def Y_iid_to_Y_iid_tilde(Y, alpha, m, n, pi_true='None'):
    if pi_true == 'None':
        #Estimate no. defective items - doesn't work well
        print("True pi not known")
    else:
        Y_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*(Y - alpha*n*pi_true)
    return Y_tilde

'''=== Transform noise matrix for AMP ==='''
def Psi_to_Psi_tilde(Psi, alpha, m):
    Psi_tilde = (1/np.sqrt(m*alpha*(1-alpha)))*Psi
    return Psi_tilde

def Y_sc_to_Y_sc_tilde(Y, alpha, W, m, n, pi_true='None'):
    """defect_no here is a length-C array where the cth element corresponds
    to the empirical distribution in the cth column-block of B_0"""
    R, C = np.shape(W)
    M = m/R
    if pi_true == 'None':
        #Estimate no. defective items - doesn't work well
        print("True pi not known")
    else:
        Y_tilde = np.zeros(m)
        for r in range(R):
            Y_tilde[r*M:(r+1)*M] = (1/np.sqrt(m*alpha*(1-alpha)/R))*(Y[r*M:(r+1)*M] - alpha*np.sum(W[r,:]*pi_true))
    return Y_tilde

def W_to_Wtilde(W, alpha):
    W_tilde = (W*(1-alpha*W))/(1-alpha)
    return W_tilde


def eta_pool(S, T_B, pi, calcJac=False):
    """
    Calculates the MMSE denoising function eta(S) = E[B | B + G = S]
    for general r>=1.

    B: p-by-L signal matrix; row-wise iid.
       Each row has a single 1, 1 is in ith column with probability pi[i]. 
    G: n-by-L Gaussian (noise) matrix; row-wise iid.
       Each row is drawn from L-dim Gaussian with mean 0 and covariance
       matrix T_B.
       Note that T_B can be obtained either empirically on the fly via
       the residual matrix Z in the AMP algo (see the amp_mmse function
       below) or from SE.
    When calcJac=True, the function also calculates the Jacobian of eta wrt
    its argument.
    The function applies row-wise to matrix inputs.

    Outputs:
        eta:    the p-by-L matrix eta(S)
        etaJac: p-by-L-by-L tensor; each L-by-L matrix is the Jacobian
                matrix corresponding to each row of S
    """
    assert np.sum(pi) == 1
    assert S.ndim == 2 and S.shape[1] == T_B.shape[0]
    p, L = S.shape
    etaJac = []  # initialise as empty
    if np.allclose(T_B, 0):  # np.array_equal(Sigma, np.zeros((r, r))):
        eta = S
        if calcJac:
            etaJac = np.tile(np.eye(L), (p, 1, 1))
    else:
        cb = np.eye(L)  # L-by-L, list of all posible codewords - implicit in code
        # Numerator:
        T_B_inv_cb = np.linalg.solve(T_B, cb)
        assert T_B_inv_cb.shape == (L, L)
        cb_T_B_inv_cb = np.sum(T_B_inv_cb.T*cb, axis=1, keepdims=True)
        assert cb_T_B_inv_cb.shape == (L, 1)
        S_T_B_inv_cb = (S @ T_B_inv_cb).T 
        assert S_T_B_inv_cb.shape == (L, p)
        pwr_arr = -0.5* (cb_T_B_inv_cb - 2 * S_T_B_inv_cb)
        max_pwr = np.max(pwr_arr, axis=0)
        pwr_minus_max_arr = pwr_arr - max_pwr
        expo = np.exp(pwr_minus_max_arr)
        pi_expo = pi[:,np.newaxis]*expo
        num = cb @ pi_expo
        assert num.shape == (L,p)
        denom = np.sum(pi_expo, axis = 0)
        assert denom.shape == (p,)
        eta = (num/denom).T
        assert eta.shape == (p, L)
        
        if calcJac:
            etaJac = etaJac_pool(pi_expo, cb, T_B_inv_cb, num, denom, L, p)

    return eta, etaJac

def etaJac_pool(pi_expo, cb, T_B_inv_cb, num, denom, L, p):
    """
    Calculates the L-by-L Jacobian matrix for S given in etaJac.
    etaJac: p-by-L-by-L
    """
    piexpo_TBinv_c = np.einsum('ij,ik->ijk', pi_expo, T_B_inv_cb.T) # dim L-p-L
    sum_piexpo_TBinv_c = np.sum(piexpo_TBinv_c, axis=0)
    sum_c_piexpo_TBinv_c = matmul_2dmat_w_3dmat(cb.T, piexpo_TBinv_c)  # dim L-p-L
    term_1 = sum_c_piexpo_TBinv_c/denom[np.newaxis, :, np.newaxis]
    assert term_1.shape == (L,p,L)
    
    term_2_num = num[:, :, np.newaxis]*sum_piexpo_TBinv_c[np.newaxis, :, :]
    assert term_2_num.shape == (L,p,L)
    term_2_den = (denom**2)[np.newaxis, :, np.newaxis]
    term_2 = term_2_num/term_2_den
    #print(term_2.shape)
    etaJac = np.transpose(term_1 - term_2, axes=(1, 0, 2))
    assert etaJac.shape == (p, L, L)

    assert np.logical_not(
        np.any(np.isnan(etaJac))
    ), "eta output from etaJac contains nans"  # no nans
    return etaJac

#STATE EVOLUTION
def se_pool(delta, L, alpha, pi, sigma, iter_max, num_mc_samples):
    """Run SE for Pooled AMP for the specified setting"""
    SigmaW = ((sigma**2)/(delta*alpha*(1-alpha)))*np.eye(L)
    MSE_mat_arr = np.zeros((iter_max, L, L))
    mse_est_array = np.zeros(iter_max)
    eff_noise_cov_arr = np.zeros((iter_max, L, L))
    correlation_arr = np.zeros(iter_max)
    signal_pwr = np.diag(pi) #analytical solution for E[B0_bar B0_bar^T]
    MSE_mat_arr[0] =  signal_pwr # L-by-L
    eff_noise_cov_arr[0] = SigmaW + (1/delta)*MSE_mat_arr[0]
    mse_est_array[0] = np.average(np.diag(MSE_mat_arr[0,:,:]))
    for t in range(1, iter_max):
        #print("se iter: ", t)
        
        Sigma = eff_noise_cov_arr[t-1] 
        
        Sigma_chol = np.linalg.cholesky(Sigma)
        G_arr = (Sigma_chol @ np.random.randn(L, num_mc_samples)).T
        X_arr = create_B(pi, num_mc_samples)
        S = X_arr + G_arr
        assert S.shape == (num_mc_samples, L)
        eta_S, _ = eta_pool(S, Sigma, pi)

        eta_sqr_arr = np.einsum('ij,ik->ijk', eta_S, eta_S)
        E_S_eta_sqr = np.mean(eta_sqr_arr, axis=0)
        
        MSE_mat_arr[t] = signal_pwr - E_S_eta_sqr
        
        eff_noise_cov_arr[t] = SigmaW + (1/delta)*MSE_mat_arr[t]
        
        mse_est_array[t] = np.average(np.diag(MSE_mat_arr[t,:,:]))
        
        correlation_arr[t] = np.mean(np.einsum('ij,ij->i', eta_S, eta_S))
        
        #Stopping criterion - Relative norm tolerance: If true, break loop, stop algorithm
        if (mse_est_array[t] - mse_est_array[t-1])**2/mse_est_array[t]**2 < 1e-6 or mse_est_array[t]<1e-50:
            mse_est_array = mse_est_array[:t+1]
            MSE_mat_arr = MSE_mat_arr[:t+1]
            eff_noise_cov_arr = eff_noise_cov_arr[:t+1]
            correlation_arr = correlation_arr[:t+1]
            break
        
    return MSE_mat_arr, eff_noise_cov_arr, mse_est_array, correlation_arr



def amp_pool(pi, A, Y, max_iter, X0):
    """
    AMP recursion to update the estimate X and the residual Z.
    The model is Y = AX + W.
    Inputs:
        alpha: prob that a row of X is all-zero
        X_amp: p-by-L; current AMP estimate of the signal matrix X
        Z_amp: n-by-L; current AMP residual matrix
        eta_etaJac_fn: function of the form:
            eta, etaJac = eta_etaJac_fn(alpha, S, Sigma, calcJac)
    This function empirically estimates Sigma, i.e. the covariance of
    the effective noise Z. An alternative way is to estimate Sigma via SE.
    """
    assert np.sum(pi) == 1
    n, p = A.shape
    L = Y.shape[1]
    assert n == Y.shape[0]
    
    scalar_mse_arr = np.zeros(max_iter)
    eff_noise_cov_arr = np.zeros((max_iter, L, L))
    
    #Initialization
    X_amp = np.zeros((p,L))#pi*np.ones((p,L))
    Z_amp = Y
    
    for t in range(max_iter):
    
        eff_noise_cov_arr[t] = np.cov(Z_amp, rowvar=False)
        Sigma = eff_noise_cov_arr[t]
        if np.min(np.linalg.eigvals(Sigma)) < 1e-10:
            Sigma +=  np.eye(L)*1e-5 #Add noise to enforce positive semi-definiteness
        # Effective noisy observation of signal matrix X:
        S = X_amp + A.T @ Z_amp
        # Update estimate of X and calculate Jacobian:
        X_amp, etaJac = eta_pool(S, Sigma, pi, calcJac=True)
        # Update residual matrix Z using the current Z and the updated X:
        Z_amp = Y - A @ X_amp + (1/n)*Z_amp @ (np.sum(etaJac, axis=0).T)
        
        scalar_mse_arr[t] = np.mean((X_amp - X0) ** 2)
        
        #Stopping criterion - Relative norm tolerance
        if (np.linalg.norm(eff_noise_cov_arr[t] - eff_noise_cov_arr[t-1], 2)/np.linalg.norm(eff_noise_cov_arr[t], 2)) < 1e-9 or np.linalg.norm(eff_noise_cov_arr[t], 2)<1e-10:
            scalar_mse_arr = scalar_mse_arr[:t+1]
            eff_noise_cov_arr = eff_noise_cov_arr[:t+1]
            break
    
    mse_final = scalar_mse_arr[t]
        
    return X_amp, Z_amp, scalar_mse_arr, eff_noise_cov_arr, mse_final


def matmul_2dmat_w_3dmat(mat_2d, mat_3d):
    """
    mat_2d has dimension d0-by-d1
    mat_3d has dimension d0-by-d2-by-d3
    Align mat_2d and mat_3d along their zeroth axis;
    calculate the outer product of each row of mat_2d with
    each layer of mat_3d (spanned between 1st and 2nd axes)
    to yield a 4D tensor;
    sum along the zeroth axis to give res_3d, of dimension d1-by-d2-by-d3.
    """
    assert mat_2d.ndim == 2 and mat_3d.ndim == 3
    assert mat_2d.shape[0] == mat_3d.shape[0]
    d1 = mat_2d.shape[1]
    d2 = mat_3d.shape[1]
    d3 = mat_3d.shape[2]
    res_3d = np.einsum("ij,ikl->jkl", mat_2d, mat_3d)
    assert res_3d.shape == (d1, d2, d3)
    return res_3d

def check_cov_mat(Sigma):
    """Check if Sigma is a valid covariance matrix."""
    if np.ndim(Sigma) == 0:  # scalar
        assert Sigma >= 0
    else:
        assert Sigma.ndim == 2
        assert Sigma.shape[0] == Sigma.shape[1]
        assert np.all(np.linalg.eigvals(Sigma) >= 0)  # +ve semidefinite
        assert np.allclose(Sigma, Sigma.T)
        
        
#Produces one-hot vector with 1 in specified index
def one_hot_vec(index, length):
  output = np.zeros(length)
  output[index] = 1
  return output

"""def category_check(matrix, estimate, sparsity_lvls):
  p = len(estimate)
  L = len(estimate[0])
  output = matrix
  for l in range(L):
    l_num_items = sparsity_lvls[l]*p
    if np.sum(estimate[:,l]) == l_num_items:
      output[:,l] = np.full(p,matrix.min())
  return output

def estimate(input, sparsity_lvls):
  p = len(input)
  L = len(input[0])
  output = np.zeros((p,L))
  matrix = input
  for iter in range(p):
    item, category = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    output[item,:] = one_hot_vec(category, L)
    matrix[item,:] = np.full(L,input.min())
    matrix = category_check(matrix,output,sparsity_lvls)
  return output

# The greedy version
def IHT_greedy(Y, X, p, sparsity_lvls, num_iter):
  L = len(sparsity_lvls)
  B_k = np.zeros((p, L))
  for k in range(num_iter):
    input = B_k + np.dot(X.T,Y-np.dot(X,B_k))
    B_k = estimate(input, sparsity_lvls)
  return B_k
"""


def category_check(matrix, estimate, sparsity_lvls, category, min_value):
  p = len(estimate)
  outpu2 = np.copy(matrix)
  l_num_items = sparsity_lvls[category]
  if np.sum(estimate[:,category]) == l_num_items:
    # remove category (columns) from further consideration
    outpu2[:,category] = np.full(p, min_value)
  return outpu2

def estimate(inpu, sparsity_lvls):
  p = len(inpu)
  L = len(inpu[0])
  outpu = np.zeros((p,L))
  min_value = inpu.min() - 100
  matrix = np.copy(inpu) #This was messing with variable inpu
 
  # first remove catergory (columns) with zero items
  for l in range(L):
    if sparsity_lvls[l] == 0:
      matrix[:,l] = np.full(p, min_value)
 
  for iter in range(p):
      #find largest entry in the input matrix - make hard decision
    item, category = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    outpu[item,:] = one_hot_vec(category, L)
    # remove item from further consideration
    matrix[item,:] = np.full(L, min_value)
    matrix = category_check(matrix, outpu, sparsity_lvls, category, min_value)
  return outpu

# Iterative Hard Thresholding (IHT) algorithm
def IHT_greedy(Y, X, p, sparsity_lvls, num_iter, B_0):
  L = len(sparsity_lvls)
  B_k = np.zeros((p, L))
  corr_array = []
  for k in range(num_iter):
    print("Num. Iteration: ", k)
    XB_k = X @ B_k
    inpu = B_k + (X.T @ (Y - XB_k))
    
    B_k = estimate(inpu, sparsity_lvls)
    print("B_k ", B_k)
    correl = np.mean(np.einsum('ij, ij->i', B_k, B_0))
    corr_array.append(correl)
  return B_k, corr_array

def run_NP_iid_Gauss(n, p, L, Y, X, var_psi, B_prop):

  # Configuring our inputs to suit CVX
  Y_opt = Y.flatten('F')
  X_opt = np.zeros((n*L,p*L))
  for l in range(L):
    X_opt[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_opt = np.eye(p)
  for l in range(L-1):
    C_opt = np.concatenate((C_opt, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective matrix and vector
  q = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    q -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = C_opt
  # Setting the equality constraints vector
  b = one_p

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Define and solve the CVXPY problem
  constant = 1/(2*var_psi)
  B_opt = cp.Variable(p*L)
  objective = cp.Minimize(constant*cp.sum_squares(Y_opt - X_opt @ B_opt) + (q.T @ B_opt))
  constraints = []
  constraints.append(G @ B_opt <= h)
  constraints.append(A @ B_opt == b)
  problem = cp.Problem(objective, constraints)
  problem.solve(solver=cp.OSQP)
  print('optimal obj value:', problem.value)

  # Reconfiguring our outputs to suit pooled data
  B_QP_est = B_opt.value
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_QP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est
