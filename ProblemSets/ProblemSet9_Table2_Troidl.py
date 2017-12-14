#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:59:24 2017

@author: alexandradinu
"""


# import packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import numba

import numpy as np
import scipy.stats as st
from scipy.stats import norm
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from numpy import inf
import ar1_approx as ar1

#Defining Parameters
rho = 0.7605
mu = 0.0
sigma_eps = 0.213
alpha_k = 0.297
alpha_l = 0.650
delta = 0.154
psi = 1.08
r = 0.04
h = 6.616
betafirm = (1 / (1 + r))
num = 9
w = 0.7
sizez = 9
num_draws = 100 #EQUAL TO T
sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))
theta = np.array((alpha_k, rho, psi, sigma_z))

##Setting up the firm problem
# Creating K grid which will be called kvec
dens = 5
kbar = 12 #kstar * 500
lb_k = 0.001
ub_k = kbar
krat = np.log(lb_k / ub_k)
numb = np.ceil(krat / np.log(1 - delta))
K = np.empty(int(numb * dens))
for j in range(int(numb * dens)):
    K[j] = ub_k * (1 - delta) ** (j / dens)
kvec = K[::-1]
sizek = kvec.shape[0]

# Grid for z and transition matrix
def z_func(theta):
    # We will use the Rouwenhorst (1995) method to approximate a continuous
    # distribution of shocks to the AR1 process with a Markov process.

    alpha_k, rho, psi, sigma_z = theta
    num_sigma = 3
    step = (num_sigma * sigma_z) / (sizez / 2)
    Pi, z = ar1.rouwen(rho, mu, step, sizez)
    Pi = np.transpose(Pi)  # make so rows are where start, columns where go
    z = np.exp(z)  # because the AR(1) process was for the log of productivity
    return z, Pi

z, Pi = z_func(theta)

#Loop to use with value function iteration
@numba.jit
def VFI(V, e, betafirm, sizez, sizek, Vmat, Pi):
    V_prime = np.dot(Pi, V)
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for k in range(sizek): # loop over k'
                Vmat[i, j, k] = e[i, j, k] + betafirm * V_prime[i, k]
    return Vmat

#Value Function Iteration
def solve_firm(theta):

    alpha_k, rho, psi, sigma_z = theta

    sizez = z.shape[0]
    op = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i,j] = z[i] * (kvec[j]**alpha_k)

    e = np.zeros((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            for k in range(sizek):
                e[i, j, k] = (op[i,j] - kvec[k] + ((1 - delta) * kvec[j]) -
                            ((psi / 2) * ((kvec[k] - ((1 - delta) * kvec[j])) ** 2)
                            / kvec[j]))


    # Value funtion iteration
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek))  # initial guess at value function
    Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1

    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        Vmat = VFI(V, e, betafirm, sizez, sizek, Vmat, Pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,)  # store value function at each
        # iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
        Vstore[:,:, i] = V  # store V at each iteration of VFI
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        VFiter += 1

    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
        print('VFI took ', VFI_time, ' seconds to solve')

    VF = V  # solution to the functional equation


    #decision rules

    ### Collect optimal values(functions)
    # Optimal capital stock k'
    optK = kvec[PF]

     # optimal investment I
    optI = optK - (1 - delta) * kvec

    return VF, PF, optK, optI

VF, PF, optK, optI= solve_firm(theta)

# Simulating the Markov process
def sim_markov(z, Pi, num_draws):
    # draw some random numbers on [0, 1]
    u = np.random.uniform(size=num_draws)

    # Do simulations
    z_discrete = np.empty(num_draws)  # this will be a vector of values
    oldind = 0
    z_discrete[0] = oldind
    for i in range(1, num_draws):
        sum_p = 0
        ind = 0
        while sum_p < u[i]:
            sum_p = sum_p + Pi[ind, oldind]
            ind += 1
        if ind > 0:
            ind -= 1
        z_discrete[i] = ind
        oldind = ind
        z_discrete = z_discrete.astype(dtype = np.int)

    return z_discrete

# Call simulation function to get simulated values
z_discrete = sim_markov(z, np.transpose(Pi), num_draws)

n = 1000 #number of firms
T = 100  #number of time periods
z_new = np.zeros((n,T), dtype = np.int)
for i in range(n):
    z_new[i] = sim_markov(z, np.transpose(Pi), num_draws)  #correct z

#Creating optimal capital, investment, profit
def loop_k():
    next_k = np.zeros((n,T), dtype = np.int)
    for i in range(n):
        for j in range(T-1):
            next_k[i, j+1] = PF[z_new[i,j]][next_k[i,j]]
    return next_k

next_k=loop_k()
next_k

def loop_I():
    next_optI = np.zeros((n,T))
    for i in range(n):
        for j in range(T-1):
            next_optI[i,j+1] = kvec[next_k[i,j+1]] - (1 - delta) * kvec[next_k[i,j]]
    return next_optI

next_optI = loop_I()
next_optI

def profit():
    profit = np.zeros((n, T))
    for i in range(n):
        for j in range(T):
            profit[i,j] = z_new[i,j] * (kvec[next_k[i,j]]**alpha_k)
    return profit

profit_1 = profit()
profit_1

@numba.jit
def VFI_sim():
    next_Vs = np.zeros((n,T), dtype = np.int)
    for i in range(n):
        for j in range(T):
            next_Vs[i,j] = VF[z_new[i,j], next_k[i,j]]
    return next_Vs

next_Vs = VFI_sim()
next_Vs

#Simulating the firm problem
def sim_firm(theta):
    alpha_k, rho, psi, sigma_z = theta
    z, Pi = z_func(theta)
    z_new = np.zeros((n,T), dtype = np.int)
    for i in range(n):
        z_new[i] = sim_markov(z, np.transpose(Pi), num_draws)  #correct z

    next_k = loop_k()
    next_optI = loop_I()
    profit_1 = profit()
    next_Vs = VFI_sim()

    return next_Vs, profit_1, next_optI, next_k

next_Vs, profit_1, next_optI, next_k = sim_firm(theta)

def moments(theta):
    alpha_k, rho, psi, sigma_z = theta

    '''This creates the moments'''

    inv_nextk = next_optI/kvec[next_k]
    invest_nextk = inv_nextk.reshape((1, n*100))  # I over K reshaped for corrcoef()
    sc_invest_nextk = np.corrcoef(invest_nextk[0][1:], invest_nextk[0][:100000-1])[0,1]
    profit_nextk = profit_1/kvec[next_k]
    sd_profit_nextk = profit_nextk.reshape((1, n*100))
    sd_profit_nextk = np.std(sd_profit_nextk)
    q_bar = next_Vs.sum()/kvec[next_k].sum()
    Y = invest_nextk
    Y = Y.reshape(100000, 1)
    q = next_Vs/kvec[next_k]
    q = q.reshape(100000, 1)
    prof_k = profit_1/kvec[next_k]
    prof_k = prof_k.reshape(100000, 1)
    cons = np.ones(100000)
    cons = cons.reshape(100000, 1)
    ###REGRESSIONS##
    x = np.hstack((cons, q, prof_k))
    trans = x.transpose()
    xx = np.dot(trans, x)
    inv = np.linalg.inv(xx)
    xy = np.dot(trans, Y)
    reg_coef = np.dot(inv, xy)
    mew_s = np.array([max(reg_coef[1]), max(reg_coef[2]), sc_invest_nextk, sd_profit_nextk, q_bar])  #Mu_s

    return mew_s


#Creating the distance function
def dist(A,B,C):
    d = np.linalg.multi_dot([np.transpose(A - B),np.linalg.inv(C),(A - B)])
    return d

## The objective function for minimization

def Qfunc(theta):
    alpha_k, rho, psi, sigma_z = theta
    a1 = 0.045  #these values are from table 2, row 4
    a2=0.24
    sc=0.04
    std=0.25
    qbar=2.96
    mu_s=moments(theta)
    mu_d=np.array([a1,a2,sc,std,qbar])
    W=np.eye(len(mu_d))
    dis=dist(mu_s,mu_d,W)
    return dis

dist(mu_s, mu_d, W)

# Setting bounds for alpha_k, rho, psi, sigma_z
bounds=((0,1),(0,0.3),(0,0.2),(0,1))

theta_hat=differential_evolution(Qfunc,bounds)

theta_hat.x #coefficients at table 2

#Setting up numerical derivative
epsilon = 0.1
#new theta with epsilon
theta_new = theta + epsilon

def Qfunc2(theta_new):
    alpha_k, rho, psi, sigma_z = theta
    a1 = 0.045  #these values are from table 2, row 4
    a2=0.24
    sc=0.04
    std=0.25
    qbar=2.96
    mu_s=moments(theta)
    mu_d=np.array([a1,a2,sc,std,qbar])
    W=np.eye(len(mu_d))
    dis=dist(mu_s,mu_d,W)
    return dis

theta_new_hat=differential_evolution(Qfunc2,bounds)

theta_new_hat.x

numerator = theta_new_hat.x - theta_hat.x
numerical_derivative = np.mat([numerator/epsilon])
standard_errors = np.dot(numerical_derivative.T,numerical_derivative)
standard_errors #should be a 4x4 matrix
np.diag(standard_errors) #standard errors
