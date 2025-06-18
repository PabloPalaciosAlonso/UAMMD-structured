import sys,os

import pyUAMMD

import math
import numpy as np
import json
import jsbeautifier

from scipy.spatial.transform import Rotation as R

with open("parameters.json", "r") as f:
    param = json.load(f)

N           = param["N"]
timeStep    = param["timeStep"]
temperature = param["temperature"]
volume      = param["volume"]
anisotropy  = param["anisotropy"]
gyroRatio   = param["gyroRatio"]
damping     = param["damping"]
msat        = param["msat"]
viscosity   = param["viscosity"]
b0          = param["b0"]
direction   = [0,0,1]

radius = (3*volume/(np.pi*4))**(1./3.)
magneticMoment = volume*msat

nSteps        = param["nSteps"]
firstStepSave = param["firstStepSave"]
nStepsOutput  = param["nStepsOutput"]
nStepsMeasure = param["nStepsMeasure"]

#Compute box size
L = param["L"]
box = [L,L,L]

#Create simulation

simulation = pyUAMMD.simulation()

simulation["system"] = {}
simulation["system"]["info"] = {}
simulation["system"]["info"]["type"] = ["Simulation","Information"]
simulation["system"]["info"]["parameters"] = {}
simulation["system"]["info"]["parameters"]["name"] = "MagneticThermalFluctuationsTest"

simulation["global"] = {}

simulation["global"]["types"] = {}
simulation["global"]["types"]["type"]   = ["Types","Basic"]
simulation["global"]["types"]["labels"] = ["name", "mass", "radius", "charge"]
simulation["global"]["types"]["data"]  = [["A", 0, radius, 0]]

simulation["global"]["ensemble"] = {}
simulation["global"]["ensemble"]["type"]   = ["Ensemble","NVT"]
simulation["global"]["ensemble"]["labels"] = ["box", "temperature"]
simulation["global"]["ensemble"]["data"]   = [[box, temperature]]


simulation["integrator"] = {}
simulation["integrator"]["LLG-Brown"] = {}
simulation["integrator"]["LLG-Brown"]["type"]                    = ["MagneticMotion", "Brownian"]
simulation["integrator"]["LLG-Brown"]["parameters"]              = {}
simulation["integrator"]["LLG-Brown"]["parameters"]["timeStep"]  = timeStep
simulation["integrator"]["LLG-Brown"]["parameters"]["msat"]      = msat
simulation["integrator"]["LLG-Brown"]["parameters"]["damping"]   = damping
simulation["integrator"]["LLG-Brown"]["parameters"]["gyroRatio"] = gyroRatio
simulation["integrator"]["LLG-Brown"]["parameters"]["viscosity"] = viscosity
simulation["integrator"]["LLG-Brown"]["parameters"]["magneticIntegrator"] = "LLG_Heun"

simulation["integrator"]["schedule"] = {}
simulation["integrator"]["schedule"]["type"] = ["Schedule", "Integrator"]
simulation["integrator"]["schedule"]["labels"] = ["order", "integrator","steps"]
simulation["integrator"]["schedule"]["data"] = [
    [1, "LLG-Brown", nSteps],
]

simulation["state"] = {}
simulation["state"]["labels"] = ["id", "position", "direction", "magnetization"]
simulation["state"]["data"] = []

for i in range(N):
    #quat = np.roll(R.random().as_quat(), 1)  # [w, x, y, z]
    quat = [1, 0, 0, 0]
    n, vx, vy, vz = quat
    simulation["state"]["data"].append([
        i,
        [0, 0, 0],
        [n, vx, vy, vz],
        [0,0,1-2*(i%2==0),magneticMoment]
    ])

simulation["topology"] = {}
simulation["topology"]["structure"] = {}
simulation["topology"]["structure"]["labels"] = ["id", "type"]
simulation["topology"]["structure"]["data"] = []
for i in range(N):
    simulation["topology"]["structure"]["data"].append([i, "A"])


simulation["topology"]["forceField"] = {}
simulation["topology"]["forceField"]["External"] = {}
simulation["topology"]["forceField"]["External"]["type"] = ["External", "ConstantMagneticField"]
simulation["topology"]["forceField"]["External"]["parameters"]              = {}
simulation["topology"]["forceField"]["External"]["parameters"]["b0"]        = b0
simulation["topology"]["forceField"]["External"]["parameters"]["direction"] = direction

simulation["topology"]["forceField"]["External2"] = {}
simulation["topology"]["forceField"]["External2"]["type"] = ["External", "UniaxialMagneticAnisotropy"]
simulation["topology"]["forceField"]["External2"]["parameters"]              = {"anisotropy":N*[anisotropy]}
#Output

simulation["simulationStep"] = {}
simulation["simulationStep"]["info"] = {}
simulation["simulationStep"]["info"]["type"] = ["UtilsStep", "InfoStep"]
simulation["simulationStep"]["info"]["parameters"] = {}
simulation["simulationStep"]["info"]["parameters"]["intervalStep"] = nStepsOutput


simulation["simulationStep"]["write"] = {}
simulation["simulationStep"]["write"]["type"] = ["WriteStep", "WriteStep"]
simulation["simulationStep"]["write"]["parameters"] = {}
simulation["simulationStep"]["write"]["parameters"]["intervalStep"] = nStepsOutput
simulation["simulationStep"]["write"]["parameters"]["outputFilePath"] = f"output_k{anisotropy}_b{b0}"
simulation["simulationStep"]["write"]["parameters"]["outputFormat"] = "magnet"
simulation["simulationStep"]["write"]["parameters"]["startStep"] = firstStepSave

simulation["simulationStep"]["writeDir"] = {}
simulation["simulationStep"]["writeDir"]["type"] = ["WriteStep", "WriteStep"]
simulation["simulationStep"]["writeDir"]["parameters"] = {}
simulation["simulationStep"]["writeDir"]["parameters"]["intervalStep"] = nStepsOutput
simulation["simulationStep"]["writeDir"]["parameters"]["outputFilePath"] = f"output_k{anisotropy}_b{b0}"
simulation["simulationStep"]["writeDir"]["parameters"]["outputFormat"] = "spo"
simulation["simulationStep"]["writeDir"]["parameters"]["startStep"] = firstStepSave

"""
simulation["simulationStep"]["eulerMaruyamaRigid"] = {}
simulation["simulationStep"]["eulerMaruyamaRigid"]["parameters"] = {}

"""
#Check if ./results folder exists, if not create it
if not os.path.exists("./results"):
    os.makedirs("./results")

simulation.write("./results/simulation.json")
