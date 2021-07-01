## adapted from https://medium.com/analytics-vidhya/practical-introduction-to-hartree-fock-448fc64c107b

import numpy as np
from numpy import *
import scipy
from scipy.special import erf
import matplotlib.pyplot as plt



def xyz_reader(xyz):
    # Reads an xyz file (https://en.wikipedia.org/wiki/XYZ_file_format) and returns the number of atoms,
    # atom types and atom coordinates.
    
    
    number_of_atoms = 0
    atom_type = []
    atom_coordinates = []
    
    for idx,line in enumerate(xyz.splitlines()):
        # Get number of atoms
        if idx == 0:
            try:
                number_of_atoms = line.split()[0]
                #print(number_of_atoms)
            except:
                print("xyz file not in correct format. Make sure the format follows: https://en.wikipedia.org/wiki/XYZ_file_format")
        
        # Skip the comment/blank line
        if idx == 1:
            continue

        # Get atom types and positions
        if idx != 0:
            split = line.split()
            atom = split[0]
            coordinates = np.array([float(split[1]),
                           float(split[2]),
                           float(split[3])])

            atom_type.append(atom)
            atom_coordinates.append(coordinates)
    
    return int(number_of_atoms), atom_type, atom_coordinates

def gauss_product(gauss_A, gauss_B):
    # The product of two Gaussians gives another Gaussian (pp411)
    # Pass in the exponent and centre as a tuple
    
    a, Ra = gauss_A
    b, Rb = gauss_B
    p = a + b
    diff = np.linalg.norm(Ra-Rb)**2             # squared difference of the two centres
    N = (4*a*b/(pi**2))**0.75                   # Normalisation
    K = N*exp(-a*b/p*diff)                      # New prefactor
    Rp = (a*Ra + b*Rb)/p                        # New centre
                     
    return p, diff, K, Rp

# Overlap integral (pp411)
def overlap(A, B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (pi/p)**1.5
    return prefactor*K

# Kinetic integral (pp412)
def kinetic(A,B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (pi/p)**1.5
    
    a, Ra = A
    b, Rb = B    
    reduced_exponent = a*b/p
    return reduced_exponent*(3-2*reduced_exponent*diff)*prefactor*K


# Fo function for calculating potential and e-e repulsion integrals.
# Just a variant of the error function
# pp414
def Fo(t):
    if t == 0:
        return 1
    else:
        return (0.5*(pi/t)**0.5)*erf(t**0.5)


# Nuclear-electron integral (pp412)
def potential(A,B,atom_idx,atom_coordinates,atoms,charge_dict):
    p,diff,K,Rp = gauss_product(A,B)
    Rc = atom_coordinates[atom_idx] # Position of atom C
    Zc = charge_dict[atoms[atom_idx]] # Charge of atom C
    
    
    return (-2*pi*Zc/p)*K*Fo(p*np.linalg.norm(Rp-Rc)**2)

# (ab|cd) integral (pp413)
def multi(A,B,C,D):
    p, diff_ab, K_ab, Rp = gauss_product(A,B)
    q, diff_cd, K_cd, Rq = gauss_product(C,D)
    multi_prefactor = 2*pi**2.5*(p*q*(p+q)**0.5)**-1
    return multi_prefactor*K_ab*K_cd*Fo(p*q/(p+q)*np.linalg.norm(Rp-Rq)**2)



def calculate_integrals(zeta_dict, max_quantum_number, D, alpha, charge_dict, N_atoms, atoms, atom_coordinates, STOnG, B, N):
    #global zeta_dict, max_quantum_number, D, alpha, charge_dict, N_atoms, atoms, atom_coordinates, STOng, B, N
    # Initialise matrices
    S = np.zeros((B,B))
    T = np.zeros((B,B))
    V = np.zeros((B,B))
    multi_electron_tensor = np.zeros((B,B,B,B))
   
    gcfl = []
    
    # Iterate through atoms
    for idx_a, val_a in enumerate(atoms):
        
        # For each atom, get the charge and centre
        Za = charge_dict[val_a]
        Ra = atom_coordinates[idx_a]
        
        # Iterate through quantum numbers (1s, 2s etc)
        for m in range(max_quantum_number[val_a]):
            
            # For each quantum number, get the contraction
            # coefficients, then get zeta,
            # then scale the exponents accordingly (pp158)
            d_vec_m = D[m]
            zeta = zeta_dict[val_a][m]
            alpha_vec_m = alpha[m]*zeta**2
            gcfl.append({val_a:[alpha_vec_m,d_vec_m]})
            # Iterate over the contraction coefficients
            for p in range(STOnG):
                
                
                # Iterate through atoms once again (more info in blog post)
                for idx_b, val_b in enumerate(atoms):
                    Zb = charge_dict[val_b]
                    Rb = atom_coordinates[idx_b]
                    for n in range(max_quantum_number[val_b]):
                        d_vec_n = D[n]
                        zeta = zeta_dict[val_b][n]
                        alpha_vec_n = alpha[n]*zeta**2
                        for q in range(STOnG):
                            
                        
                            # This indexing is explained in the blog post.
                            # In short, it is due to Python indexing
                            # starting at 0.
                            
                            a = (idx_a+1)*(m+1)-1
                            b = (idx_b+1)*(n+1)-1
                            
                            # Generate the overlap, kinetic and potential matrices
                            
                            S[a,b] += d_vec_m[p]*d_vec_n[q]*overlap((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb))
                            T[a,b] += d_vec_m[p]*d_vec_n[q]*kinetic((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb))
    
                            for i in range(N_atoms):
                                V[a,b] += d_vec_m[p]*d_vec_n[q]*potential((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb),i,atom_coordinates,atoms,charge_dict)
                                
                                
                            # 2 more iterations to get the multi-electron-tensor
                            for idx_c, val_c in enumerate(atoms):
                                Zc = charge_dict[val_c]
                                Rc = atom_coordinates[idx_c]
                                for k in range(max_quantum_number[val_c]):
                                    d_vec_k = D[k]
                                    zeta = zeta_dict[val_c][k]
                                    alpha_vec_k = alpha[k]*zeta**2
                                    for r in range(STOnG):
                                        for idx_d, val_d in enumerate(atoms):
                                            Zd = charge_dict[val_d]
                                            Rd = atom_coordinates[idx_d]
                                            for l in range(max_quantum_number[val_d]):
                                                d_vec_l = D[l]
                                                zeta = zeta_dict[val_d][l]
                                                alpha_vec_l = alpha[l]*zeta**2
                                                for s in range(STOnG):
                                                    c = (idx_c+1)*(k+1)-1
                                                    d = (idx_d+1)*(l+1)-1
                                                    multi_electron_tensor[a,b,c,d] += d_vec_m[p]*d_vec_n[q]*d_vec_k[r]*d_vec_l[s]*(
                                                    multi((alpha_vec_m[p],Ra),
                                                          (alpha_vec_n[q],Rb),
                                                          (alpha_vec_k[r],Rc),
                                                          (alpha_vec_l[s],Rd))
                                                    )
    
    return S,T,V,multi_electron_tensor,gcfl


def orthoganalize_basis(S):

    evalS, U = np.linalg.eig(S)
    diagS = dot(U.T,dot(S,U))
    diagS_minushalf = diag(diagonal(diagS)**-0.5)
    X = dot(U,dot(diagS_minushalf,U.T))

    return(X)




def SD_successive_density_matrix_elements(Ptilde,P,B):
    x = 0
    for i in range(B):
        for j in range(B):
            x += B**-2*(Ptilde[i,j]-P[i,j])**2
    
    return x**0.5

def get_coeffs_and_eigenvalues(Fock, X):
    # Calculate Fock matrix in orthogonalised base
    Fockprime = dot(X.T,dot(Fock, X))
    evalFockprime, Cprime = np.linalg.eig(Fockprime)

    #Correct ordering of eigenvalues and eigenvectors (starting from ground MO as first column of C, else we get the wrong P)
    idx = evalFockprime.argsort()
    evalFockprime = evalFockprime[idx]
    Cprime = Cprime[:,idx]

    C = dot(X,Cprime) 

    return C, evalFockprime

def g1(alpha,r,Ra):
    return np.exp(-alpha*np.linalg.norm(r-Ra)**2)

def CGF(a,d,r,Ra):
    #print(g1(a[0],r,Ra))
    return d[0]*g1(a[0],r,Ra) + d[1]*g1(a[1],r,Ra) + d[2]*g1(a[2],r,Ra)

def dens(a1,a2,d,r,Ra,Rb,P):

    return P[0,0]*CGF(a1,d,r,Ra)*CGF(a1,d,r,Ra)+ P[1,1]*CGF(a2,d,r,Rb)*CGF(a2,d,r,Rb)+2*P[0,1]*CGF(a1,d,r,Ra)*CGF(a2,d,r,Rb)

def orb_antibond(a1,a2,d,r,Ra,Rb,C):

    return -C[0,1]*CGF(a1,d,r,Ra)+ -C[1,1]*CGF(a2,d,r,Rb)

def orb_bond(a1,a2,d,r,Ra,Rb,C):

    return -C[0,0]*CGF(a1,d,r,Ra)+ -C[1,0]*CGF(a2,d,r,Rb)


def plot_dens(a1,a2,dvec,Ra,Rb,P):

    yq, xq = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

    #z = dens(a1,a2,dvec,np.array([x,y,0]),Ra,Rb,P)
    #z = dens(a1,a2,dvec,np.array([x,0,y]),Ra,Rb,P)
    zq = dens(a2,a1,dvec,np.array([0,xq,yq]),Ra,Rb,P)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    zq = zq[:-1, :-1]

    fig, ax = plt.subplots()

    c = ax.pcolormesh(xq, yq, zq, cmap='RdBu', vmin=0, vmax=1.5)
    ax.set_title('HeH+ density')
    # set the limits of the plot to the limits of the data
    ax.axis([xq.min(), xq.max(), yq.min(), yq.max()])
    fig.colorbar(c, ax=ax)

    plt.show()


def plot_orb_bond(a1,a2,dvec,Ra,Rb,C):

    yq, xq = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

    #z = dens(a1,a2,dvec,np.array([x,y,0]),Ra,Rb,P)
    #z = dens(a1,a2,dvec,np.array([x,0,y]),Ra,Rb,P)
    zq = orb_bond(a2,a1,dvec,np.array([0,xq,yq]),Ra,Rb,C)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    zq = zq[:-1, :-1]

    fig, ax = plt.subplots()

    c = ax.pcolormesh(xq, yq, zq, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('HeH+ bonding orbital')
    # set the limits of the plot to the limits of the data
    ax.axis([xq.min(), xq.max(), yq.min(), yq.max()])
    fig.colorbar(c, ax=ax)

    plt.show()


def plot_orb_antibond(a1,a2,dvec,Ra,Rb,C):

    yq, xq = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

    #z = dens(a1,a2,dvec,np.array([x,y,0]),Ra,Rb,P)
    #z = dens(a1,a2,dvec,np.array([x,0,y]),Ra,Rb,P)
    zq = orb_antibond(a1,a2,dvec,np.array([0,xq,yq]),Ra,Rb,C)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    zq = zq[:-1, :-1]

    fig, ax = plt.subplots()

    c = ax.pcolormesh(xq, yq, zq, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('HeH+ antibonding orbital')
    # set the limits of the plot to the limits of the data
    ax.axis([xq.min(), xq.max(), yq.min(), yq.max()])
    fig.colorbar(c, ax=ax)

    plt.show()


