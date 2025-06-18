import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import quad

def compute_energy(ctheta, b0, m0):
    return -m0 * b0 * ctheta
    
def integrand(ctheta, b0, m0, kBT):
    energy = compute_energy(ctheta, b0, m0)
    return np.exp(-energy / kBT)

def probabilityDensity(ctheta, b0, m0, kBT):
    Z, _ = quad(
        integrand,
        -1, 1,
        args=(b0, m0, kBT)
    )

    energy            = compute_energy(ctheta, b0, m0)
    prob_unnormalized = np.exp(-energy / kBT)
    prob_normalized = prob_unnormalized / Z

    return prob_normalized

with open("parameters.json", "r") as f:
    param = json.load(f)

temperature = param["temperature"]
volume      = param["volume"]
anisotropy  = param["anisotropy"]
b0          = param["b0"]
msat        = param["msat"]
initStep    = 1000

m0 = volume*msat

dataSimulation = np.loadtxt(f"./results/output_k{anisotropy}_b{b0}.magnet")[initStep:,:]
cthetaSim      = dataSimulation[:,-1]/np.sqrt(np.dot(dataSimulation[0,:],dataSimulation[0,:]))

cthetaTheo = np.linspace(-1,1)
P          = probabilityDensity(cthetaTheo, b0, m0, temperature)

plt.figure(figsize=(6,5))
plt.hist(cthetaSim, density = True, bins = 200)
plt.plot(cthetaTheo, P)
plt.show()
