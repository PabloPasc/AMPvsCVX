#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:01:21 2023

@author: pp423
"""

from pool_amp import run_NP_iid_Gauss, se_pool, amp_pool, eta_pool, create_B, X_iid_to_X_tilde, Y_iid_to_Y_iid_tilde, run_LP, run_NP, IHT_greedy
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
import cvxpy as cp



def amp_pool_with_init(pi, A, Y, max_iter, X0, X_init):
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
    
    #Initialization - with specified estimate X_init
    X_amp = X_init#np.zeros((p,L))#pi*np.ones((p,L))
    # Use Z_amp=Y OR Z_amp = Y - A@X_amp
    Z_amp = Y - A@X_amp
    
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


#-------------------Fig. 2b - sigma=0.1------------------------------------------
#Simulations
alpha = 0.5
run_no = 10
sigma = 0.1
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.linspace(0.1, 0.6, num=6)
delta_se_array = np.linspace(0.1, 0.6, 51)

#Green plot
pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

delta_corr_amp_cvx = []
delta_corr_std_amp_cvx = []
delta_corr_amp_true_init = []
delta_corr_std_amp_true_init = []

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp_cvx = []
    delta_corr_runs_amp_true_init = []
    
    
    for run in range(run_no):
        
        print("Run: ", run)
        
        B_0 = create_B(pi, p)
        X = np.random.normal(0,1/np.sqrt(n),size=(n,p))
        var_psi = (sigma**2)/(delta*alpha*(1-alpha))
        Psi = np.random.normal(0, np.sqrt(var_psi), (n,L))
        Y = np.dot(X, B_0) + Psi
        
        #NP - with pi
        B_NP_est = run_NP_iid_Gauss(n, p, L, Y, X, var_psi, pi)
        
        
        
        #AMP - on iid Gaussian matrix - with CVX estimate as Initializer
               
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool_with_init(pi, X, Y, max_iter, B_0, B_NP_est)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp_cvx.append(corr_av_b)
        
        
        #AMP - on iid Gaussian matrix - with true signal B_0 as Initializer
               
        B, _, mse_arr, noise_arr, mse_final_b2 = amp_pool_with_init(pi, X, Y, max_iter, B_0, B_0)
        corr_av_b2 = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b2)
        delta_corr_runs_amp_true_init.append(corr_av_b2)
        
        
        
    
    delta_corr_amp_cvx.append(np.mean(delta_corr_runs_amp_cvx))
    delta_corr_std_amp_cvx.append(np.std(delta_corr_runs_amp_cvx))
    delta_corr_amp_true_init.append(np.mean(delta_corr_runs_amp_true_init))
    delta_corr_std_amp_true_init.append(np.std(delta_corr_runs_amp_true_init))
    
    
plt.figure()
plt.errorbar(delta_array, delta_corr_amp_cvx, yerr=delta_corr_std_amp_cvx, label =r"CVX + AMP", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_amp_true_init, yerr=delta_corr_std_amp_true_init, label =r"True Signal + AMP", fmt='*', color='black',ecolor='lightgrey', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
tikzplotlib.save("check2_fig2b.tex")

#-------------------Fig. 2c - sigma=0.3------------------------------------------
#Simulations
alpha = 0.5
run_no = 10
sigma = 0.3
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.linspace(0.1, 1.7, num=17)
delta_se_array = np.linspace(0.1, 1.7, 161)

#Green plot
pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

delta_corr_amp_cvx = []
delta_corr_std_amp_cvx = []
delta_corr_amp_true_init = []
delta_corr_std_amp_true_init = []

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp_cvx = []
    delta_corr_runs_amp_true_init = []
    
    
    for run in range(run_no):
        
        print("Run: ", run)
        
        B_0 = create_B(pi, p)
        X = np.random.normal(0,1/np.sqrt(n),size=(n,p))
        var_psi = (sigma**2)/(delta*alpha*(1-alpha))
        Psi = np.random.normal(0, np.sqrt(var_psi), (n,L))
        Y = np.dot(X, B_0) + Psi
        
        #NP - with pi
        B_NP_est = run_NP_iid_Gauss(n, p, L, Y, X, var_psi, pi)
        
        
        
        #AMP - on iid Gaussian matrix - with CVX estimate as Initializer
               
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool_with_init(pi, X, Y, max_iter, B_0, B_NP_est)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp_cvx.append(corr_av_b)
        
        
        #AMP - on iid Gaussian matrix - with true signal B_0 as Initializer
               
        B, _, mse_arr, noise_arr, mse_final_b2 = amp_pool_with_init(pi, X, Y, max_iter, B_0, B_0)
        corr_av_b2 = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b2)
        delta_corr_runs_amp_true_init.append(corr_av_b2)
        
        
        
    
    delta_corr_amp_cvx.append(np.mean(delta_corr_runs_amp_cvx))
    delta_corr_std_amp_cvx.append(np.std(delta_corr_runs_amp_cvx))
    delta_corr_amp_true_init.append(np.mean(delta_corr_runs_amp_true_init))
    delta_corr_std_amp_true_init.append(np.std(delta_corr_runs_amp_true_init))
    
    
plt.figure()
plt.errorbar(delta_array, delta_corr_amp_cvx, yerr=delta_corr_std_amp_cvx, label =r"CVX + AMP", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_amp_true_init, yerr=delta_corr_std_amp_true_init, label =r"True Signal + AMP", fmt='*', color='black',ecolor='lightgrey', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("check2_fig2c.tex")