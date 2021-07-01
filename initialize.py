# Basis set variables
import numpy as np
# STO-nG (number of gaussians used to form a contracted gaussian orbital - pp153)
#STOnG = 3

global zeta_dict, max_quantum_number, D, alpha, charge_dict

# Dictionary of zeta values (pp159-160, 170)
zeta_dict = {'H': [1.24], 'He':[2.0925], 'Li':[2.69,0.80],'Be':[3.68,1.15],
            'B':[4.68,1.50],'C':[5.67,1.72]}   #Put zeta number in list to accomodate for possibly more basis sets (eg 2s orbital)

# Dictionary containing the max quantum number of each atom, 
# for a minimal basis STO-nG calculation
max_quantum_number = {'H':1,'He':1,'Li':2,'Be':2,'C':2}


# Gaussian contraction coefficients (pp157)
# Going up to 2s orbital (W. J. Hehre, R. F. Stewart, and J. A. Pople. J. Chem. Phys. 51, 2657 (1969))
# Row represents 1s, 2s etc...
D = np.array([[0.444635, 0.535328, 0.154329],
              [0.700115,0.399513,-0.0999672]])

# Gaussian orbital exponents (pp153)
# Going up to 2s orbital (W. J. Hehre, R. F. Stewart, and J. A. Pople. J. Chem. Phys. 51, 2657 (1969))
alpha = np.array([[0.109818, 0.405771, 2.22766],
                     [0.0751386,0.231031,0.994203]])

# Basis set size
#B = 0
#for atom in atoms:
#    B += max_quantum_number[atom]

# Other book-keeping

# Number of electrons (Important!!)
#N = 2

# Keep a dictionary of charges
charge_dict = {'H': 1, 'He': 2, 'Li':3, 'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10}
