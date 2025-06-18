import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import quad

def compute_energy(cphi, K, V):
    return K*V*(1-cphi**2)
    
def integrand(cphi, K, V, kBT):
    energy = compute_energy(cphi, K, V)
    return np.exp(-energy / kBT)

def probabilityDensity(cphi, K, V, kBT):
    Z, _ = quad(
        integrand,
        -1, 1,
        args=(K, V, kBT)
    )

    energy            = compute_energy(cphi, K, V)
    prob_unnormalized = np.exp(-energy / kBT)
    prob_normalized = prob_unnormalized / Z

    return prob_normalized

def computeCosThetaSim(dataSimulation):
    return dataSimulation[:,-1]/np.sqrt(np.dot(dataSimulation[0,:],dataSimulation[0,:]))

def computeCosPhiSim(dataSimulation, posSimulation, initStep):
    indices = np.arange(0, len(posSimu), 4)
    uk      = posSimu[indices + 3] - posSimu[indices]
    uk      = uk[initStep:]

    dot_products = np.sum(dataSimulation * uk, axis=1)
    norms_data   = np.linalg.norm(dataSimulation, axis=1)
    norms_uk     = np.linalg.norm(uk, axis=1)
    cphi         = dot_products / (norms_data * norms_uk)

    return cphi


with open("parameters.json", "r") as f:
    param = json.load(f)

temperature = param["temperature"]
volume      = param["volume"]
anisotropy  = param["anisotropy"]
b0          = param["b0"]
msat        = param["msat"]
initStep    = 1000

m0 = volume*msat
cphiTheo = np.linspace(-1,1)

dataSimulation = np.loadtxt(f"./results/output_k{anisotropy}_b{b0}.magnet")[initStep:,:]
posSimu        = np.loadtxt(f"./results/output_k{anisotropy}_b{b0}.spo")[:,:3]
cPhiSim        = computeCosPhiSim(dataSimulation, posSimu, initStep)
P              = probabilityDensity(cphiTheo, anisotropy, volume, temperature)

plt.figure(figsize=(6,5))
plt.hist(cPhiSim, density = True, bins = 200)
plt.plot(cphiTheo, P)
plt.show()
