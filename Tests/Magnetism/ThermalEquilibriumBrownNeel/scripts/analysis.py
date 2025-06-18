import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import dblquad


def compute_energy(ctheta, cphi, K, V, b0, m0):
    eanis  = K * V * (1 - cphi**2)
    efield = - m0 * b0 * ctheta
    return eanis + efield

# Función escalar para la densidad de probabilidad no normalizada (con jacobiano sin(theta))
def integrand(ctheta, cphi, K, V, b0, m0, kBT):
    energy = compute_energy(ctheta, cphi, K, V, b0, m0)
    return np.exp(-energy / kBT)

# Función principal para calcular la densidad de probabilidad normalizada en un punto
def probabilityDensity(ctheta_vals, cphi_vals, K, V, b0, m0, kBT):
    # Normalización exacta por integración doble
    Z, _ = dblquad(
        integrand,
        -1, 1,             # theta bounds
        lambda theta: -1,      # phi lower bound
        lambda theta: 1, # phi upper bound
        args=(K, V, b0, m0, kBT)
    )

    # Crear mallas para evaluación vectorizada
    cTheta, cPhi = np.meshgrid(ctheta_vals, cphi_vals, indexing='ij')
    Energy       = compute_energy(cTheta, cPhi, K, V, b0, m0)
    prob_unnormalized = np.exp(-Energy / kBT)
    prob_normalized = prob_unnormalized / Z

    return prob_normalized

with open("parameters.json", "r") as f:
    param = json.load(f)

temperature = param["temperature"]
volume      = param["volume"]
anisotropy  = param["anisotropy"]
b0          = param["b0"]
msat        = param["msat"]

m0 = volume*msat

initStep = 1000
#Read results of the simulation
dataSimulation = np.loadtxt(f"./results/output_k{anisotropy}.magnet")[initStep:,:]
posSimu        = np.loadtxt(f"./results/output_k{anisotropy}.spo")[:,:3]

indices = np.arange(0, len(posSimu) - 3, 4)
uk      = posSimu[indices + 3] - posSimu[indices]

uk = uk[initStep:]

cphiSim   = np.sum(dataSimulation * uk, axis=1)
cthetaSim = dataSimulation[:,-1]/np.sqrt(np.dot(dataSimulation[0,:],dataSimulation[0,:]))

# Definir los bordes (usá mismos que en la teoría para comparar)
bins = 50
range_ = [[-1, 1], [-1, 1]]  # para ctheta y cphi

H, xedges, yedges = np.histogram2d(cthetaSim, cphiSim, bins=bins, range=range_, density=True)

# Coordenadas de centro de los bins
xcenters = 0.5 * (xedges[:-1] + xedges[1:])
ycenters = 0.5 * (yedges[:-1] + yedges[1:])
X, Y = np.meshgrid(xcenters, ycenters, indexing='ij')
plt.figure(figsize=(6,5))
plt.contourf(X, Y, H, levels=50, cmap='viridis')
plt.xlabel("cos(θ)")
plt.ylabel("cos(φ)")
plt.title("Distribución simulada: P(cos(θ), cos(φ))")
plt.colorbar(label="Densidad")
plt.tight_layout()

# Espacios de cos(θ) y cos(φ)
ctheta = np.linspace(-1, 1, 50)  # cos(θ) ∈ [-1, 1]
cphi   = np.linspace(-1, 1, 50)  # cos(φ) ∈ [-1, 1]

# Crear grilla 2D
Ctheta, Cphi = np.meshgrid(ctheta, cphi, indexing='ij')

# Calcular densidad de probabilidad normalizada
P = probabilityDensity(ctheta, cphi, anisotropy, volume, b0, m0, temperature)

# Graficar
plt.figure(figsize=(6,5))
cp = plt.contourf(Ctheta, Cphi, P, levels=50, cmap='viridis')
plt.xlabel("cos(θ)")
plt.ylabel("cos(φ)")
plt.title("Distribución de Boltzmann P(cos(θ), cos(φ))")
plt.colorbar(cp, label="Probabilidad")
plt.tight_layout()

#plt.show()



# Promedio sobre columnas (cphi) → densidad marginal en ctheta
Ptheta_sim = np.trapz(H, ycenters, axis=1)
Ptheta_sim /= np.trapz(Ptheta_sim, xcenters)  # normalizar

Ptheta_theory = np.trapz(P, cphi, axis=1)
Ptheta_theory /= np.trapz(Ptheta_theory, ctheta)  # normalizar

plt.figure(figsize=(6,4))
plt.plot(xcenters, Ptheta_sim, 'o-', label='Simulación (prom φ)')
plt.plot(ctheta, Ptheta_theory, '-', label='Teoría (prom φ)')
plt.hist(cthetaSim, density = True, bins = 50)
plt.xlabel("cos(θ)")
plt.ylabel("Densidad marginal")
plt.title("Distribución marginal en cos(θ)")
plt.legend()
plt.tight_layout()
plt.show()
