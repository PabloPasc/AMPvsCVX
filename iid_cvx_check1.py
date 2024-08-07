#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:01:21 2023

@author: pp423
"""

from pool_amp import se_pool, amp_pool, create_B, X_iid_to_X_tilde, Y_iid_to_Y_iid_tilde, run_LP, run_NP, IHT_greedy
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
import cvxpy as cp

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

delta_se_corr_amp = []
delta_corr_amp = []
delta_corr_std_amp = []
delta_corr_amp_nopi = []
delta_corr_std_amp_nopi = []
delta_corr_np = []
delta_corr_std_np = []
delta_corr_iht = []
delta_corr_std_iht = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_amp.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp = []
    delta_corr_runs_amp_nopi = []
    delta_corr_runs_np = []
    delta_corr_runs_iht = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on iid Gaussian matrix
        X = np.random.normal(0,1/np.sqrt(n),size=(n,p))
        var_psi = (sigma**2)/(delta*alpha*(1-alpha))
        Psi = np.random.normal(0, np.sqrt(var_psi), (n,L))
        Y = np.dot(X, B_0) + Psi
        
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X, Y, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp.append(corr_av_b)
        
        #NP - with pi
        B_NP_est = run_NP_iid_Gauss(n, p, L, Y, X, var_psi, pi)
        corr_av_np = np.mean(np.einsum('ij, ij->i', B_NP_est, B_0))
        delta_corr_runs_np.append(corr_av_np)
        
        
        
    
    delta_corr_amp.append(np.mean(delta_corr_runs_amp))
    delta_corr_std_amp.append(np.std(delta_corr_runs_amp))
    delta_corr_np.append(np.mean(delta_corr_runs_np))
    delta_corr_std_np.append(np.std(delta_corr_runs_np))
    
    
plt.figure()
plt.plot(delta_se_array, delta_se_corr_amp, label='SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_amp, yerr=delta_corr_std_amp, label =r"AMP, $\hat{\pi}$", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
#plt.errorbar(delta_array, delta_corr_amp_nopi, yerr=delta_corr_std_amp_nopi, label =r"AMP, $\pi$", fmt='*', color='black',ecolor='lightgrey', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_np, yerr=delta_corr_std_np, label =r"CVX", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
#plt.errorbar(delta_array, delta_corr_iht, yerr=delta_corr_std_iht, label =r"IHT", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
tikzplotlib.save("check1_fig2b.tex")

#-------------------Fig. 2c - sigma=0.3------------------------------------------
#Simulations
alpha = 0.5
run_no = 100
sigma = 0.3
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.linspace(0.1, 1.7, num=17)
delta_se_array = np.linspace(0.1, 1.7, 161)

#Green plot
pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

delta_se_corr_amp = []
delta_corr_amp = []
delta_corr_std_amp = []
delta_corr_amp_nopi = []
delta_corr_std_amp_nopi = []
delta_corr_np = []
delta_corr_std_np = []
delta_corr_iht = []
delta_corr_std_iht = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_amp.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp = []
    delta_corr_runs_amp_nopi = []
    delta_corr_runs_np = []
    delta_corr_runs_iht = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on iid Gaussian matrix
        X = np.random.normal(0,1/np.sqrt(n),size=(n,p))
        var_psi = (sigma**2)/(delta*alpha*(1-alpha))
        Psi = np.random.normal(0, np.sqrt(var_psi), (n,L))
        Y = np.dot(X, B_0) + Psi
        
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X, Y, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp.append(corr_av_b)
        
        #NP - with pi
        B_NP_est = run_NP_iid_Gauss(n, p, L, Y, X, var_psi, pi)
        corr_av_np = np.mean(np.einsum('ij, ij->i', B_NP_est, B_0))
        delta_corr_runs_np.append(corr_av_np)
        
        
        
    
    delta_corr_amp.append(np.mean(delta_corr_runs_amp))
    delta_corr_std_amp.append(np.std(delta_corr_runs_amp))
    delta_corr_np.append(np.mean(delta_corr_runs_np))
    delta_corr_std_np.append(np.std(delta_corr_runs_np))
    
    
plt.figure()
plt.plot(delta_se_array, delta_se_corr_amp, label='SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_amp, yerr=delta_corr_std_amp, label =r"AMP, $\hat{\pi}$", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
#plt.errorbar(delta_array, delta_corr_amp_nopi, yerr=delta_corr_std_amp_nopi, label =r"AMP, $\pi$", fmt='*', color='black',ecolor='lightgrey', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_np, yerr=delta_corr_std_np, label =r"CVX", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
#plt.errorbar(delta_array, delta_corr_iht, yerr=delta_corr_std_iht, label =r"IHT", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
tikzplotlib.save("check1_fig2c.tex")