'''
This file codes the answers for the Problem Set 1
'''

from stochastic_process import Stochastic_Process
import matplotlib.pyplot as plt

# Set parameters 
n, m, mu, rho, sigma = 9, 3, 0, 0.95, 0.007
T = 10000

# Exercise 1 - Tauchen's Method
ar = Stochastic_Process(n, m, mu, rho, sigma)
grid, P= ar.Tauchen()

# Exercise 2 - Rouwenhorst's Method
ar = Stochastic_Process(n, m, mu, rho, sigma)
grid, P = ar.Rouwenhorst()


'''
We can compare our transition to the one obtained using the QuantEcon
package. Junt uncomment lines 25-27
'''
#import quantecon as qe
#P_tauchen_qe = qe.markov.approximation.rouwenhorst(n, 0, sigma, rho)
#P_rouwen_qe = qe.markov.approximation.tauchen(rho, sigma, b=0.0, m=3, n=n)


# Exercise 3 - Simulate AR(1) 
# Tauchen
simulated_tauchen, simulated_continuous = ar.Simulate_AR(T)

plt.plot(simulated_continuous, label='AR(1)')
plt.plot(simulated_tauchen, label='Tauchen')
plt.legend()
plt.title('Simulation - AR(1) vs. Tauchen')
plt.grid()
plt.savefig('tauchen.png')
plt.show()

# Rouwenhorst
simulated_rouwen, _ = ar.Simulate_AR(T, tauchen=0)

plt.plot(simulated_continuous, label='AR(1)')
plt.plot(simulated_rouwen, label='Rouwenhorst')
plt.legend()
plt.title('Simulation - AR(1) vs. Rouwenhorst')
plt.grid()
plt.savefig('rouwen.png')
plt.show()


# MSE
MSE_tauchen = (simulated_tauchen-simulated_continuous)**2
MSE_rouwen = (simulated_rouwen-simulated_continuous)**2

plt.plot(MSE_tauchen, label='MSE Tauchen')
plt.plot(MSE_rouwen, label='MSE Rouwenhorst')
plt.legend()
plt.title('MSE - Tauchen vs. Rouwenhorst')
plt.grid()
plt.savefig('mse.png')
plt.show()

# Exercise 4 - rho-->1
rw = Stochastic_Process(n, m, mu, 0.99, sigma)

simulated_tauchen_rw, simulated_continuous_rw = rw.Simulate_AR(T)
simulated_rouwen_rw, _ = rw.Simulate_AR(T, tauchen=0)

plt.plot(simulated_continuous_rw, label='AR(1)')
plt.plot(simulated_tauchen_rw, label='Tauchen')
plt.legend()
plt.title('Simulation - AR(1) vs. Tauchen (ρ=0.99)')
plt.grid()
plt.savefig('tauchen_rw.png')
plt.show()

plt.plot(simulated_continuous_rw, label='AR(1)')
plt.plot(simulated_rouwen_rw, label='Rouwenhorst')
plt.legend()
plt.title('Simulation - AR(1) vs. Rouwenhorst (ρ=0.99)')
plt.grid()
plt.savefig('rouwen_rw.png')
plt.show()

# MSE
MSE_tauchen_rw = (simulated_tauchen_rw-simulated_continuous_rw)**2
MSE_rouwen_rw = (simulated_rouwen_rw-simulated_continuous_rw)**2

plt.plot(MSE_tauchen_rw, label='MSE Tauchen')
plt.plot(MSE_rouwen_rw, label='MSE Rouwenhorst')
plt.legend()
plt.title('MSE - Tauchen vs. Rouwenhorst (ρ=0.99)')
plt.grid()
plt.savefig('mse_rw.png')
plt.show()

plt.plot(simulated_tauchen_rw, label='Tauchen')
plt.plot(simulated_rouwen_rw, label='Rouwenhorst')
plt.legend()
plt.title('Tauchen vs. Rouwenhorst (ρ=0.99)')
plt.grid()
plt.savefig('tauchen_rouwen_rw.png')
plt.show()

