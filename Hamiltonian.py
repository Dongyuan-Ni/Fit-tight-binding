import pickle
import numpy as np
from numpy.linalg import *
from pythtb import *
from pymatgen.io.cif import Structure
import spglib
from pymatgen.core.structure import IStructure
from pymatgen.core.bonds import CovalentBond
from pymatgen.core.sites import Site
from pymatgen.core.sites import PeriodicSite
import matplotlib.pyplot as plt
import csv
import pickle

def Hamiltonian(t_1, kpoint):
    with open('eqbonds.pickle', 'rb') as f:
        eq_bonds = pickle.load(f)
    # print('# of equivalent bond types: {}'.format(len(eq_bonds) - 1))
    ################################ Testing Parameters ################################
    # Hopping parameters
    t1 = t_1
    ################################ Testing Parameters ################################
    t = [t1]
    struct = Structure.from_file('POSCAR.vasp')
    coords = struct.frac_coords
    # tight-binding model
    H = np.zeros((len(coords), len(coords)), dtype=complex)
    # define hopping between orbitals
    for i in range(1, len(eq_bonds)):
        for j in range(1, len(eq_bonds[i])):
            if isinstance(eq_bonds[i][j][0], int) and isinstance(eq_bonds[i][j][1], int):
                H[eq_bonds[i][j][0] - 1, eq_bonds[i][j][1] - 1] += t[i - 1]
            if isinstance(eq_bonds[i][j][0], int) and isinstance(eq_bonds[i][j][1], tuple):
                H[eq_bonds[i][j][0] - 1, eq_bonds[i][j][1][0] - 1] += t[i - 1]\
                * np.exp(2 * np.pi * 1j * kpoint.dot(eq_bonds[i][j][1][1]))
            if isinstance(eq_bonds[i][j][0], tuple) and isinstance(eq_bonds[i][j][1], int):
                H[eq_bonds[i][j][1] - 1, eq_bonds[i][j][0][0] - 1] += t[i - 1]\
                * np.exp(-2 * np.pi * 1j * kpoint.dot(eq_bonds[i][j][0][1]))
    return H + H.T.conj()
