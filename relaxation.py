#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import sys


# In[ ]:





# In[ ]:





# In[139]:





# # Source code

# In[91]:


D = 3
dirs = np.eye(D) 


# ## Potentials

# In[41]:


# Use Lennard-Jones for the pair potential
def LJ_pot(x, eps=1, sig=1, pow1=12):
    pow2 = pow1//2
    return 4*eps*((sig/x)**pow1 - (sig/x)**pow2)

# Use Lennard-Jones for the pair potential
def LJ_force(x, eps=1, sig=1, pow1=12):
    pow2 = pow1//2
    return -4*eps/sig*(-pow1*(sig/x)**(pow1+1) + pow2*(sig/x)**(pow2+1))

def LJ_f_eq_0(eps=1,sig=1,pow1=12):
    pow2 = pow1//2
    return sig*2**(1/pow2)
    


# In[ ]:





# ## Class constructor

# In[3]:


class nanoparticle:
    def __init__(self):
        return 


# ## Lattice structure

# In[33]:


def set_gold_lattice_structure(self):
    # Set an FCC structure and the lattice parameters
    
    self.a0 = 4.0          # [nm] length of the FCC cubic cell
    self.a = self.a0/np.sqrt(2) # [nm] interatomic distance
    self.D = 3

    
    self.a1 = np.array([0,1,1])*self.a0/2
    self.a2 = np.array([1,0,1])*self.a0/2
    self.a3 = np.array([1,1,0])*self.a0/2

    self.A = np.zeros([D,D])
    self.A[0,:] = self.a1
    self.A[1,:] = self.a2
    self.A[2,:] = self.a3
    
    # First nearest neighbours, in units of the primitive vectors, and corresponding coordinates
    self.first_neighs = np.array([ [1,0,0],  [0,1,0],  [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1],
                         [1,-1,0], [-1,1,0], [1,0,-1], [-1,0,1], [0,1,-1], [0,-1,1]], dtype=int)
    
    self.second_neighs = np.array([[1,1,-1], [-1,-1,1], [1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1]], dtype=int)

    # return 0

nanoparticle.set_gold_lattice_structure = set_gold_lattice_structure


# # Position generation

# In[9]:


def create_rectangular(self, N1, N2, N3):
  # Creates a nanoparticle with parallel facets

    self.N = N1*N2*N3 # total number of points
    self.coordinates = np.array([self.N, self.D])
    
    n = 0
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                R = i*self.a1 + j*self.a2 + k*self.a3
                
                self.coordinates[n,:] = [i,j,k]
                n += 1


    self.positions = self.coordinates@self.A
    self.xs = self.positions[:,0]
    self.ys = self.positions[:,1]
    self.zs = self.positions[:,2]

def create_sphere(self, radius):
    # Creates a sphere centered at the origin. The neighbours are expressed in
    # terms of the primitive vectors. a1, a2, a3 and radius are in nanometers
    coordinates = [[0,0,0]]
    
    n = 0
    while n < len(coordinates):
        a,b,c = coordinates[n] # lattice coordinates of the current atom
        
        for neigh in self.first_neighs:
            i,j,k = neigh          # difference in lattice coordinates
            x,y,z = i+a, j+b, k+c  # lattice coordinates of the new proposed atom
            R = x*self.a1 + y*self.a2 + z*self.a3 # position vector
        
            # Check if the new proposed atom is inside the sphere and that it has
            # not yet been considered
            if np.linalg.norm(R) < radius and [x,y,z] not in coordinates:
                coordinates.append([x,y,z])
        
        n += 1

    self.N = len(coordinates)
    self.coordinates = np.array(coordinates)
    self.positions = self.coordinates@self.A
    self.xs = self.positions[:,0]
    self.ys = self.positions[:,1]
    self.zs = self.positions[:,2]

nanoparticle.create_rectangular = create_rectangular
nanoparticle.create_sphere = create_sphere


# ## Neighbour list

# In[44]:


def get_neighbour_list(self, increments):
    # increments is the list of neighbours to check. This is chosen by the user manually

    max_neighs = len(increments)
    
    self.neigh_list = np.zeros([self.N,  max_neighs], dtype=int) - 1
    self.num_neigh_list = np.zeros(self.N, dtype=int)

    clist = [list(c) for c in self.coordinates]
    for n, coord in enumerate(self.coordinates):
        
        num = 0
        for inc in increments:
            neigh = list(coord + inc)
            
            if neigh in clist:
                m = clist.index(neigh)
                self.neigh_list[n, num] = m
                num += 1

        self.num_neigh_list[n] = num
            


def get_neighbour_list_automatic(self, first=True, second=False):
    if first and not second:
        self.get_neighbour_list(self.first_neighs)
    elif second and first:
        both = np.array(list(self.first_neighs)+list(self.second_neighs))
        self.get_neighbour_list(both)
    elif not first and second:
        self.get_neighbour_list(self.second_neighs)
    else:
        print("Error: No list of neighbours selected")

nanoparticle.get_neighbour_list = get_neighbour_list
nanoparticle.get_neighbour_list_automatic = get_neighbour_list_automatic


# # Dynamics

# In[53]:


def get_forces(self):
    # Assumes the force is central

    forces = np.zeros([self.N, self.D])
    for n, pos in enumerate(self.positions):

        total_dif = np.zeros(D) # normal to surface

        num_neighs = self.num_neigh_list[n]
        for m in range(num_neighs):
            neigh = self.neigh_list[n,m]
            pos_neigh = self.positions[neigh]
            dif = pos_neigh - pos
            total_dif += dif
       
            dist = np.linalg.norm(dif)
            unit = dif/dist
       
            forces[n] += -unit*self.force(dist)
    
    
        # Check if it's at the surface
        if num_neighs != 12:
            forces[n] += total_dif*self.hydrostatic


    return forces


nanoparticle.get_forces = get_forces
# def get_force(n, m, positions, neigh_list, F, hydrostatic):
#     # get the force on atom n due to atom m and hydrostatic pressure
#     pos1 = positions[n]
#     pos2 = positions[m]

#     dif = pos2 - pos1    
#     dist = np.linalg.norm(dif)
#     unit = dif/dist
#     force = -unit*F(dist)

#     # Check if it's at the surface
#     total_dif = np.zeros(D)
#     for k in neigh_list[n]:
#         pos_neigh = positions[k]
#         dif = pos_neigh - pos1
#         total_dif += dif
        
#     if len(neigh_list[n]) != 12:
#         print("Atom not in bulk")
#         force += total_dif*hydrostatic
        
#     return force


# In[66]:


def get_next_positions(self, dt):
    # updates the positions of the atoms using the function F
    # returns the new positions and the norm of the forces
    
    forces = self.get_forces()
    new_positions = self.positions + forces*dt
    dif = np.linalg.norm(forces)
    return new_positions, dif

nanoparticle.get_next_positions = get_next_positions



# In[73]:


def relax(self, niter=100, dt=0.01):
    difs = np.zeros(niter)
    
    for i in range(niter):
        new_positions, dif = self.get_next_positions(dt)
        self.positions = new_positions.copy()
        difs[i] = dif
        
    return difs

nanoparticle.relax = relax


# In[ ]:





# ## Linearization

# In[101]:


# Dynamical matrix DxD block

def get_phi_block(R1, R2, ϕ, eps = 1e-6):
    
    d0 = np.linalg.norm(R2-R1)
    
    # find the first and second derivatives
    der1 = np.zeros(D)
    der2_od = np.zeros([D,D])
    
    for i in range(D):
        u = dirs[i]*eps
        d1 = np.linalg.norm(R2 + u - R1)
        d2 = np.linalg.norm(R2 - u - R1)
        der1[i] = (ϕ(d1) - ϕ(d2))/eps/2
        der2_od[i,i] = (ϕ(d1) + ϕ(d2) - 2*ϕ(d0))/eps**2/2
    
    
    for i in range(D):
        for j in range(D):
            if i==j: continue
            uij = eps*(dirs[i] + dirs[j])
            d1 = np.linalg.norm(R2 + uij - R1)
            d2 = np.linalg.norm(R2 - uij - R1)
            der2_od[i,j] = (ϕ(d1) + ϕ(d2) - 2*ϕ(d0))/eps**2/2
            der2_od[i,j] -= der2_od[i,i] + der2_od[j,j]
    
            der2_od[i,j] /= 2
    
    return der1, der2_od*2



# In[104]:


def get_phi_matrix(self, eps = 1e-6):
    phi_matrix = np.zeros([self.N*self.D, self.N*self.D])
    for i,Ri in enumerate(self.positions):
        num_neighs = self.num_neigh_list[i]
        
        for neigh in range(num_neighs):
            j = self.neigh_list[i,neigh]
            Rj = self.positions[j]
            
            der1, der2 = get_phi_block(Ri, Rj, self.potential, eps)
            phi_matrix[i*D:i*D+D,j*D:j*D+D] = der2
    return phi_matrix

def get_dyn_matrix(self, eps = 1e-6):
    phi_matrix = self.get_phi_matrix(eps)
    dyn_matrix = -phi_matrix
    for i in range(self.N):
        for j in range(self.N):
            dyn_matrix[i*D:i*D+D, i*D:i*D+D] += phi_matrix[i*D:i*D+D,j*D:j*D+D]
    return dyn_matrix

nanoparticle.get_phi_matrix = get_phi_matrix
nanoparticle.get_dyn_matrix = get_dyn_matrix


# In[ ]:





# ## A and B force constants

# In[107]:


def get_AB_from_block(R1, R2, block):
    dR = R2 - R1
    uR = dR/np.linalg.norm(dR)

    # get two vectors orthogonal to dR
    orth_vec1 = np.array([    0, dR[2], -dR[1]])
    orth_vec2 = np.array([dR[2],     0, -dR[0]])

    # It can happen than the vector being used to orthogonalize dR
    # is along dR. So we need to try two different vectors and see
    # which one is best
    orth_vec = orth_vec1.copy()
    if np.linalg.norm(orth_vec1) < np.linalg.norm(orth_vec2):
        orth_vec = orth_vec2.copy()

    orth_vec = orth_vec/np.linalg.norm(orth_vec)
    A = orth_vec@block@orth_vec

    B_plus_A = uR@block@uR
    B = B_plus_A - A
    
    return A, B

def get_AB(self, eps=1e-6):
    A_list = self.neigh_list*0.0
    B_list = self.neigh_list*0.0
    A_matrix = np.zeros([self.N, self.N])
    B_matrix = np.zeros([self.N, self.N])
    
    for i,Ri in enumerate(self.positions):
        num_neighs = self.num_neigh_list[i]
        
        for n in range(num_neighs):
            j = self.neigh_list[i,n]
            Rj = self.positions[j]
            
            der1, block = get_phi_block(Ri, Rj, self.potential, eps)
            A, B = get_AB_from_block(Ri, Rj, block)

            A_list[i, n] = A
            B_list[i, n] = B

            A_matrix[i,j] = A
            B_matrix[i,j] = B
            
    return A_list, B_list, A_matrix, B_matrix


nanoparticle.get_AB = get_AB


# In[ ]:





# In[ ]:





# In[ ]:





# # Testing 1

# In[ ]:





# In[ ]:





# In[ ]:





# # Test 2

# In[119]:





# In[120]:





# In[86]:





# In[87]:





# In[ ]:





# In[ ]:





# In[113]:





# In[108]:





# In[114]:





# In[ ]:





# In[ ]:





# In[130]:





# In[ ]:




