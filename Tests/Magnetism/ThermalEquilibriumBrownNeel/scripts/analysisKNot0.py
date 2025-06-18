import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import quad, dblquad

def compute_energy(ctheta, cphi, b0, m0, K, V):
    return -m0 * b0 * ctheta + K*V*(1-cphi**2)
    
def integrand(ctheta, cphi, b0, m0, K, V, kBT):
    energy = compute_energy(ctheta, cphi, b0, m0, K, V)
    return np.exp(-energy / kBT)

def probabilityDensity(ctheta, cphi, b0, m0, K, V, kBT):
    Z, _ = dblquad(
        integrand,
        -1, 1,
        lambda ct:-1, lambda ct:1,
        args=(b0, m0, K, V, kBT)
    )

    energy            = compute_energy(ctheta, cphi, b0, m0, K, V)
    prob_unnormalized = np.exp(-energy / kBT)
    prob_normalized = prob_unnormalized / Z

    return prob_normalized

def computeCosThetaSim(dataSimulation):
    return dataSimulation[:,-1]/np.sqrt(np.dot(dataSimulation[0,:],dataSimulation[0,:]))

def computeCosPhiSim(dataSimulation, posSimulation, initStep):
    indices = np.arange(0, len(posSimu) - 3, 4)
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
m0          = volume*msat
initStep    = 10000

#Numerical results

dataSimulation = np.loadtxt(f"./results/output_k{anisotropy}_b{b0}.magnet")[initStep:,:]
posSimu        = np.loadtxt(f"./results/output_k{anisotropy}_b{b0}.spo")[:,:3]

cThetaSim = computeCosThetaSim(dataSimulation)
cPhiSim   = computeCosPhiSim(dataSimulation, posSimu, initStep)

plt.figure()
plt.hist2d(cThetaSim, cPhiSim, bins=100, cmap='viridis', density = True)
plt.colorbar(label="Densidad")

#Theory
cthetaTheo     = np.linspace(-1,1, 2000)
cphiTheo       = np.linspace(-1,1, 2000)
cThetaT, cPhiT = np.meshgrid(cthetaTheo, cphiTheo, indexing='ij')
P              = probabilityDensity(cThetaT, cPhiT,  b0, m0, anisotropy, volume, temperature)

plt.figure()
plt.contourf(cThetaT, cPhiT, P, levels = 50)
plt.colorbar(label="Densidad")

#Difference

H, xedges, yedges = np.histogram2d(cThetaSim, cPhiSim,
                                   bins=50, range=[[-1, 1], [-1, 1]],
                                   density=True)

xc     = 0.5*(xedges[:-1] + xedges[1:])
yc     = 0.5*(yedges[:-1] + yedges[1:])
XC, YC = np.meshgrid(xc, yc, indexing='ij')
P      = probabilityDensity(XC, YC,  b0, m0, anisotropy, volume, temperature)

diff = P - H
plt.figure()
plt.contourf(XC, YC, np.abs(diff), levels = 30)
plt.colorbar(label="Densidad")
plt.show()


