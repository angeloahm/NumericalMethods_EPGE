"""
First we create an instance that codes Tauchen/Rouwenhorst methods, and solve all the 
simulations required for given parameter.
"""

import numpy as np
from numba import jit, prange    
from scipy.stats import norm
import random


class Stochastic_Process(object):
    def __init__(self, n, m, mu, rho, sigma):
        self.n = n             #Number of gridpoints
        self.m = m             #Scale parameter
        self.mu = mu           #Drift
        self.rho = rho         #AR parameter
        self.sigma = sigma     #Variance
        #self.T = T            #Number of periods for the simulation
    
    
    @jit
    def Tauchen(self):
        
        ub = self.m*self.sigma/((1-self.rho**2)**(1/2)) #Define upper bound
        lb = -ub                                        #Define lower bound
            
        grid = np.linspace(lb, ub, self.n)              #Grid with equidistant points
            
        delta = (ub-lb)/(self.n-1)                      #Distance between two points
            
        P = np.zeros((self.n, self.n))
            
        for i in prange(self.n):   
            for j in prange(self.n):
                x_upper = (grid[j]+delta/2-(1-self.rho)*self.mu-self.rho*grid[i])/self.sigma
                x_lower = (grid[j]-delta/2-(1-self.rho)*self.mu-self.rho*grid[i])/self.sigma
                    
                P[i,j] = norm.cdf(x_upper)-norm.cdf(x_lower)
                
                x_0 = (grid[0]+delta/2-(1-self.rho)*self.mu-self.rho*grid[i])/self.sigma
                P[i,0] = norm.cdf(x_0)
                
                x_n = (grid[self.n-1]-delta/2-(1-self.rho)*self.mu-self.rho*grid[i])/self.sigma
                P[i,self.n-1] = 1-norm.cdf(x_n)
            
 
        return [grid, P]
    
    @jit
    def Rouwenhorst(self):
        
        sigma_theta = self.sigma / ((1-self.rho**2)**(1/2))
        
        ub = sigma_theta*(self.n-1)**(1/2)
        lb = -ub
        
        grid = np.linspace(lb, ub, self.n)
        
        self.p = (1+self.rho)/2
        #P_2 = np.array([(self.p, 1-self.p), (1-self.p, self.p)])
        
        C = []
        
        C.append(0)
        C.append(1)
        
        
        for i in range(self.n-1):
            P1 = np.zeros(shape=(i+2,i+2))
            P2 = np.zeros(shape=(i+2,i+2))
            P3 = np.zeros(shape=(i+2,i+2))
            P4 = np.zeros(shape=(i+2,i+2))
            
            P1[:i+1, :i+1] = self.p*C[i+1]
            P2[:i+1, 1:] = (1-self.p)*C[i+1]
            P3[1:, :-1] = (1-self.p)*C[i+1]
            P4[1:, 1:] = self.p*C[i+1]            
            
            C.append(P1 + P2 + P3 + P4)

        P = C[self.n]
        
        for i in range(self.n):
            P[i,:] = P[i,:]/np.sum(P[i,:])
        
        return [grid, P]
    
    # This function simulates the AR(1) process using Tauchen/Rouwenhorst and compares it
    # to the continous version
    @jit
    def Simulate_AR(self, T, tauchen=1):
        
        random.seed(49)
        
        #Allocate arrays
        simulated_continuous = np.empty(T)
        simulated_discrete = np.empty(T)
        
        simulated_continuous[0] = 0             #Set initial value
        
        #Tauchen's flag
        if tauchen==1:
            grid, P = self.Tauchen()
            start = np.where(grid==0)
            state = start
        else:
            grid, P = self.Rouwenhorst()
            start = np.where(grid==0)
            state = start
                
        # Start simulation
        for t in prange(T-1):
            
            # Pick a normal shock
            shock = np.random.normal()

            #Continuous simulation
            simulated_continuous[t+1] = (1-self.rho)*self.mu+self.rho*simulated_continuous[t]+self.sigma*shock
            
            #Maping normal shocks to uniform 
            cum_sum = np.cumsum(P[state,:])
            state = np.sum(norm.cdf(shock)>=cum_sum)
            
            simulated_discrete[t+1] = grid[state]
            
        return simulated_discrete, simulated_continuous

            
        
        
    











