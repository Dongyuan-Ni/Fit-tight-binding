from __future__ import print_function
from pythtb import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import *
import csv
from Hamiltonian import *

###################### INPUT #########################
noccupancy = 4
path=[[0.,0.,0.],[0.5,0.0,0.0],[.333,.333,0.0],[0.,0.,0.]]
kpoints = []
for i in range(len(path) - 1):
    kpoints.append(np.linspace(path[i], path[i+1], 100))
kpoints = np.array(kpoints).reshape(-1, 3)
Num_bands = 2
Num_kpoints = 300
###################### INPUT #########################

################## Collect PBE data ###################
data_PBE = []
kdistance_PBE = []
energy_PBE = []
with open('pbe.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in list(csv_reader):
        data_PBE.append(row)
data_PBE = data_PBE[:-3]
kdistance_tmp = []
energy_tmp = []
for i in range(len(data_PBE)):
    if len(data_PBE[i][0].split()) != 1:
        kdistance_tmp.append(float(data_PBE[i][0].split()[0]))
        energy_tmp.append(float(data_PBE[i][0].split()[1]))
    else:
        kdistance_PBE.append(kdistance_tmp)
        energy_PBE.append(energy_tmp)
        kdistance_tmp = []
        energy_tmp = []
for i in range(len(energy_PBE)):
    for j in range(len(energy_PBE[i])):
        energy_PBE[i][j] -= -3.2541
PBE_ref = np.array(energy_PBE[noccupancy-1:noccupancy+1])
# plot PBE bandstructure
fig, ax = plt.subplots()
for i in range(len(energy_PBE)):
    ax.plot(kdistance_PBE[i], energy_PBE[i], c='gray')
################## Collect PBE data ###################

################# optimization ####################
learning_rate = 1e-3
t = -1 # initialize t
for epoch in range(1000):
    loss = 0
    delta = 0
    # momentum term
    vt = 0
    for idx in range(len(kpoints)):
        kpoint = kpoints[idx]
        H = Hamiltonian(t, kpoints[idx])
        eigenvalues, eigenvectors = eigh(H)
        eigenvalues_matrix = np.diag(eigenvalues)
        eigenvectors_left = eigenvectors
        eigenvectors_right = inv(eigenvectors)
        for nband in range(Num_bands):
            loss += (eigenvalues[nband] - PBE_ref[nband, idx])**2
        ################# gradient ################
        Ek_bar = (1/Num_bands) * (1/Num_kpoints) * np.diag([(eigenvalues[0] - PBE_ref[0, idx]), (eigenvalues[1] - PBE_ref[1, idx])])
        delta = eigenvectors_left.dot(Ek_bar).dot(eigenvectors_right)
        vt = 0.8 * vt - learning_rate * delta[0, 1]
        t += vt
        ################# gradient ################


    if epoch % 100 == 0:
        print('Epoch: {}; Loss: {}; Parameters: {}'.format(epoch, loss, t))
################# optimization ####################

# plot tb bandstructure
# t = -2.5
energy_tb = []
for idx in range(len(kpoints)):
    H = Hamiltonian(t, kpoints[idx])
    eigenvalues_tb, eigenvectors_tb = eigh(H)
    energy_tb.append(eigenvalues_tb)
energy_tb = np.array(energy_tb).T
for i in range(len(energy_tb)):
    ax.plot(kdistance_PBE[i], energy_tb[i], c='red')
# ax.set_ylim(-3, 3)
plt.show()
