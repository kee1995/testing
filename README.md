# testing

import numpy as np
from scipy import constants 
import matplotlib.pyplot as plt
import sys
import os
import meshio
from dolfin import*
#Domain 
from mshr import*

import time

t1=time.perf_counter()


mesh = UnitSquareMesh(5,5)
#domain=Rectangle(Point(0,0), Point(1,1))
#mesh= generate_mesh(domain, 1)
plot(mesh)
plt.show()

#defining functionspace
order=1

V=FunctionSpace(mesh, 'CG', order)
u = TrialFunction(V)
v = TestFunction(V)


print("number of nodes", mesh.num_vertices())
#print("number of edges", mesh.num_edges())
print("number of elements", mesh.num_cells())

a=(inner(grad(u),grad(v)))*dx
b=inner(u,v)*dx

if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

A = PETScMatrix() #stiffness matrix
assemble(a, tensor=A)

B=PETScMatrix() #mass matrix
assemble(b, tensor=B)

print(type(A.array()))
#print("stiffness matrix",A.array())
#print("mass matrix", B.array())

B_inv=np.linalg.inv(B.array())
#print("inverse of B",B_inv)

C=np.dot(B_inv, A.array())

#print("B_inv*A", C)

lam, vec=np.linalg.eig(C)
#print(lam)

print("maximum eigenvalue", max(lam))
print("maximum eigenvalue", min(lam))

