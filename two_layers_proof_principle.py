#%% IMPORT
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import datetime
from matplotlib.animation import FuncAnimation
from scipy import integrate

import scipy as scipy
import logging
import os

logger = logging.getLogger(__name__)

#%% SIMULATION PARAMETERS

'SIMULATION PARAMETERS'
Lx = 5 #length of the simulation box
Nx = 2**9
Ly = 0.01
timestepper = d3.RK443 #time iteration scheme
timestep = 5e-5#5e-4 # timestep
stop_time = 1#5 # max simulating time
dealias = 3/2 # aliasing
N_save = 500 # number of save

h = 1e-3 # numerical help
r = 1*30  # Stabilization with diffusion coefficient r

# position of the filaments
xl_A = -1
xr_A = 1
xl_B = -1.6
xr_B = 0.4
# generic values -1.5 0.5 -0,5 1.5


# Phase field model 
D_f = 0.2
li = 0.2 # length of the filaments interface
G_f = 1/18*li**2


'PHYSICAL PARAMETERS'
#Temperature
kT = 1

# Parameter related to the density of binding sites
n_s = 500#125
# binding energies
B = 10 # Energy wall
Eb_A = -1
Eb_B = -1
Eb_D = -4

# Diffusion
D_A = 1
D_B = 1
D_D = 1

 # Chemical reaction coefficients
L_1 = 0.1
L_2 = 0.1
L_3 = 0.1
L_4 = 0.1

# viscosity 
m_s = 1 # viscosity 
m = 1
# Elastic properties
G_s = 1 # relaxation of the stress
G = 10
E_s = 1 # Young modulus
E = 3
K_s = 1   # conversion between deviatoric stress and elastic stress
K = 1

gamma = 1000 #friction coef

'__________________________________________________________'

# stop_time = 5/f # Setting the maximun time to have 5 periodes
# if stop_time <= 1*1/L:
#     stop_time = 1*1/L

# stop_time = 1


# Setting the name of the files
dir_name = os.path.dirname(__file__)
name_save_file = "test-4-dec"
name_save = dir_name + "/"+ name_save_file

extension = '.mp4'
# print(name_save)


#%% DEDALUS BASIS, FUNCTION DEFINITION AND DEDALUS FIELD
# BUILDING DEDALUS COORDINATES AND BASIS
coords = d3.CartesianCoordinates('x','y')

dtype = np.float64
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)

x = dist.local_grids(xbasis)
ex, ey = coords.unit_vector_fields(dist)

# DEFINING FUNCTIONS AND SUBTITUTION
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

Lap = lambda A: d3.Laplacian(A)
grad = lambda A: d3.Gradient(A)
div = lambda A: d3.Divergence(A)

T = lambda A: d3.TransposeComponents(A)
Tr = lambda A: d3.Trace(A)
S = lambda A: 0.5*( A + T(A) )
A = lambda A: 0.5*( A - T(A) )

grad_S = lambda A: 0.5*( grad(A) + T(grad(A)) )
grad_S_BC = lambda A: 0.5*(dx(A@ex) + dy(A@ey))
grad_A = lambda A: 0.5*( grad(A) - T(grad(A)) )

Dx = lambda v,G: v@grad(G) + grad_A(v)@G + T(T(G)@grad_A(v)) # Co rotational convection derivative
ct = lambda A,B: d3.Trace(A@d3.TransposeComponents(B)) # Used for T_pq \partial_alpha T_pq

Ln = lambda A: np.log(A)


def filament_function(F,xl,xr,x,li):
    F = 1/(1+np.exp(-6/li*(x-xl))) * 1/(1+np.exp(6/li*(x-xr)))
    return F


def func_step(v,t_s,tx,l_s,t):
    # t_s : time of jump
    # tx : moment of the jump
    # t : time array
    # l_s : amplitude of the jump
    v = l_s/t_s*np.exp((t-tx)/t_s)/(1+np.exp((t-tx)/t_s))**2
    return v


# SETTING DEDALUS FIELDS
f_A = dist.Field(name='f_A',bases=(xbasis))
f_B = dist.Field(name='f_B',bases=(xbasis))
f_D = dist.Field(name='f_D',bases=(xbasis))


E_A = dist.Field(name='E_A',bases=(xbasis))
U_A = dist.Field(name='U_A',bases=(xbasis))
E_B = dist.Field(name='E_B',bases=(xbasis))
E_D = dist.Field(name='E_D',bases=(xbasis))

n_A = dist.Field(name='n_A',bases=(xbasis))
n_B = dist.Field(name='n_B',bases=(xbasis))
n_D = dist.Field(name='n_D',bases=(xbasis))

# Equilibirum densities
D_eq = dist.Field(name='D_eq',bases=(xbasis))
D_eq_A = dist.Field(name='D_eq_A',bases=(xbasis))
A_eq = dist.Field(name='A_eq',bases=(xbasis))
D_eq_B = dist.Field(name='D_eq_B',bases=(xbasis))
B_eq = dist.Field(name='B_eq',bases=(xbasis))



# Velocities
V_A = dist.Field(name='V_A',bases=(xbasis))
V_B = dist.Field(name='V_B',bases=(xbasis))
V_M = dist.Field(name='V_D',bases=(xbasis))

# Elastic stress
u_el_A_xx = dist.Field(name='u_el_A_xy',bases=(xbasis))
u_el_A_xy = dist.Field(name='u_el_A_xy',bases=(xbasis))
u_el_A_yx = dist.Field(name='u_el_A_yx',bases=(xbasis))
u_el_A_yy = dist.Field(name='u_el_A_yy',bases=(xbasis))

u_el_B_xx = dist.Field(name='u_el_B_xy',bases=(xbasis))
u_el_B_xy = dist.Field(name='u_el_B_xy',bases=(xbasis))
u_el_B_yx = dist.Field(name='u_el_B_yx',bases=(xbasis))
u_el_B_yy = dist.Field(name='u_el_B_yy',bases=(xbasis))


f_ent = dist.Field(name='f_ent',bases=xbasis)

F_A = dist.Field(name='f_A',bases=xbasis)
F_B = dist.Field(name='f_B',bases=xbasis)

force_B = dist.Field(name='force_B',bases=xbasis)
force_viscous_B = dist.Field(name='force_viscous_B',bases=xbasis)
force_elastic_B = dist.Field(name='force_elastic_B',bases=xbasis)



force_S = dist.Field(name='force_S',bases=xbasis)
u_el_A_xy = dist.Field(name='u_el_A_xy',bases=(xbasis))
# u_el_M = dist.TensorField(coords,name='u_el_A',bases=(xbasis))
u_el_B_xy = dist.Field(name='u_el_B_xy',bases=(xbasis))
u_el_xy = dist.Field(name='u_el_xy',bases=(xbasis))



# %% EQUATIONS OF THE PROBLEM
problem = d3.IVP([ f_A,f_B, n_A,n_B,n_D, E_A,E_B,E_D, A_eq,B_eq,D_eq_A,D_eq_B, V_M,V_A,V_B, f_ent ,u_el_A_xx,u_el_A_xy,u_el_A_yx,u_el_A_yy,u_el_B_xx,u_el_B_xy,u_el_B_yx,u_el_B_yy,  force_B, force_elastic_B,force_viscous_B], namespace=locals()) # Declaration of the problem variables


# Phase field of the filaments
problem.add_equation("dt(f_A) +D_f*Lap(-2*f_A + G_f*Lap(f_A)) = D_f*Lap(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A) ")
problem.add_equation("dt(f_B) +D_f*Lap(-2*f_B + G_f*Lap(f_B)) = D_f*Lap(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
                   
# Particles densities  
problem.add_equation("dt(n_A) -D_A*Lap(n_A) -r*D_A*Lap(n_A) +L_1*n_A -L_3*n_D  = D_A*dx(n_A*(1-n_D-n_A)/(1-n_D)*dx(E_A)) +D_A*dx(n_A/(1-n_D)*dx(n_D)) -r*D_A*dx(dx(n_A)) -dx(n_A*V_A)  +L_1*A_eq -L_3*D_eq_A")
problem.add_equation("dt(n_B) -D_B*Lap(n_B) -r*D_B*Lap(n_B) +L_2*n_B -L_4*n_D  = D_B*dx(n_B*(1-n_D-n_B)/(1-n_D)*dx(E_B)) +D_B*dx(n_B/(1-n_D)*dx(n_D)) -r*D_B*dx(dx(n_B)) -dx(n_B*V_B)  +L_2*B_eq -L_4*D_eq_B")
problem.add_equation("dt(n_D) -D_D*Lap(n_D) -r*D_D*Lap(n_D) +L_3*n_D +L_4*n_D  = D_D*dx(1/(1/n_D-1/(1-n_D)+1/(1-n_D-n_A)+1/(1-n_D-n_B))*dx(E_D)) +D_D*dx(1/(1-n_D-n_A)*1/(1/n_D-1/(1-n_D)+1/(1-n_D-n_A)+1/(1-n_D-n_B))*dx(n_A)) +D_D*dx(1/(1-n_D-n_B)*1/(1/n_D-1/(1-n_D)+1/(1-n_D-n_A)+1/(1-n_D-n_B))*dx(n_B)) -r*D_D*dx(dx(n_D)) -dx(n_D*V_M) +L_3*D_eq_A +L_4*D_eq_B") 

# Energy landscapes
# problem.add_equation("E_A = B*(1-f_A) + Eb_A*f_A")
# problem.add_equation("E_B = B*(1-f_B) + Eb_B*f_B")
# problem.add_equation("E_D = B*(1-f_D) + Eb_D*f_D")
#test
problem.add_equation("E_A = -np.log( f_A/(1-f_A + np.exp(Eb_A)) + np.exp(-B) )")
problem.add_equation("E_B = -np.log( f_B/(1-f_B + np.exp(Eb_B)) + np.exp(-B) )")
problem.add_equation("E_D = -np.log( f_D/(1-f_D + np.exp(Eb_D)) + np.exp(-B) )")


# Particles chemical reactions
# problem.add_equation("A_eq = (1-n_D)/(1+np.exp(E_A))")
# problem.add_equation("B_eq = (1-n_D)/(1+np.exp(E_B))")
# problem.add_equation("D_eq_A = 0.5* ( 1+n_A*np.exp(E_A-E_D) -np.sqrt( (1+n_A*np.exp(E_A-E_D))**2 -4*n_A*(1-n_B)*np.exp(E_A-E_D)  )  )")
# problem.add_equation("D_eq_B = 0.5* ( 1+n_B*np.exp(E_B-E_D) -np.sqrt( (1+n_B*np.exp(E_B-E_D))**2 -4*n_B*(1-n_A)*np.exp(E_B-E_D)  )  )")

#test
problem.add_equation("A_eq = (f_A-n_D)/(1+np.exp(Eb_A))")
problem.add_equation("B_eq = (f_B-n_D)/(1+np.exp(Eb_B))")
problem.add_equation("D_eq_A = f_D*0.5* ( 1+n_A*np.exp(Eb_A-Eb_D) -np.sqrt( (1+n_A*np.exp(Eb_A-Eb_D))**2 -4*n_A*(1-n_B)*np.exp(Eb_A-Eb_D)  )  )")
problem.add_equation("D_eq_B = f_D*0.5* ( 1+n_B*np.exp(Eb_B-Eb_D) -np.sqrt( (1+n_B*np.exp(Eb_B-Eb_D))**2 -4*n_B*(1-n_A)*np.exp(Eb_B-Eb_D)  )  )")


#Elasticity
problem.add_equation("dt(u_el_A_xx) +G*u_el_A_xx   = -V_A*dx(u_el_A_xx) -1/Ly*(V_M-V_A)*u_el_A_yx -1/Ly*(V_M-V_A)*u_el_A_xy +K*n_D*dx(V_A) ")
problem.add_equation("dt(u_el_A_xy) +G*u_el_A_xy   = -V_A*dx(u_el_A_xy) -1/Ly*(V_M-V_A)*u_el_A_yy +1/Ly*(V_M-V_A)*u_el_A_xx +K*n_D/Ly*(V_M-V_A)")
problem.add_equation("dt(u_el_A_yx) +G*u_el_A_yx   = -V_A*dx(u_el_A_yx) +1/Ly*(V_M-V_A)*u_el_A_xx -1/Ly*(V_M-V_A)*u_el_A_yy +K*n_D/Ly*(V_M-V_A)")
problem.add_equation("dt(u_el_A_yy) +G*u_el_A_yy   = -V_A*dx(u_el_A_yy) +1/Ly*(V_M-V_A)*u_el_A_xy +1/Ly*(V_M-V_A)*u_el_A_yx ")

problem.add_equation("dt(u_el_B_xx) +G*u_el_B_xx   = -V_B*dx(u_el_B_xx) -1/Ly*(V_B-V_M)*u_el_B_yx -1/Ly*(V_B-V_M)*u_el_B_xy +K*n_D*dx(V_B) ")
problem.add_equation("dt(u_el_B_xy) +G*u_el_B_xy   = -V_B*dx(u_el_B_xy) -1/Ly*(V_B-V_M)*u_el_B_yy +1/Ly*(V_B-V_M)*u_el_B_xx +K*n_D/Ly*(V_B-V_M)")
problem.add_equation("dt(u_el_B_yx) +G*u_el_B_yx   = -V_B*dx(u_el_B_yx) +1/Ly*(V_B-V_M)*u_el_B_xx -1/Ly*(V_B-V_M)*u_el_B_yy +K*n_D/Ly*(V_B-V_M)")
problem.add_equation("dt(u_el_B_yy) +G*u_el_B_yy   = -V_B*dx(u_el_B_yy) +1/Ly*(V_B-V_M)*u_el_B_xy +1/Ly*(V_B-V_M)*u_el_B_yx ")


# Force and velocity calculat
problem.add_equation("force_B = d3.Integrate(f_B*( Ly/4*(2*f_ent + m*dx(V_A*f_A+V_B*f_B) +dx(E*u_el_A_xx+E*u_el_B_xx))   +E*(2*u_el_B_xy-E*u_el_A_xy)) ,('x'))")
problem.add_equation("force_viscous_B = d3.Integrate(f_B*(Ly/4*(2*f_ent + m*dx(V_A*f_A+V_B*f_B) +dx(E*u_el_A_xx+E*u_el_B_xx))) ,('x'))")
problem.add_equation("force_elastic_B = d3.Integrate(f_B*E*(2*u_el_B_xy-u_el_A_xy) ,('x'))")


problem.add_equation("f_ent = -n_s*(n_A*dx(E_A) +n_B*dx(E_B) +n_D*dx(E_D)  +n_A*(1/(1-n_D-n_A+h))*dx(n_A)*kT  +n_B*(1/(1-n_D-n_B+h))*dx(n_B)*kT +((1-2*n_D)/(1-n_D+h) + (n_D+n_A)/(1-n_D-n_A+h) + (n_D+n_B)/(1-n_D-n_B+h))*dx(n_D)*kT)")

# Veloctity field

problem.add_equation("V_A = 0")
problem.add_equation("V_M = 0.5*(V_A*f_A+V_B*f_B) + (Ly**2)/(4*m)*(2*f_ent + m*dx(V_A+V_B) +dx(E*u_el_A_xx+E*u_el_B_xx)) + E*Ly/m*(u_el_B_xy - u_el_A_xy)  ")
# problem.add_equation("V_B*(gamma+m/(Ly*2)) = d3.Integrate(f_B*( Ly/4*(2*f_ent + m*dx(V_A*f_A+V_B*f_B) +dx(E*u_el_A_xx+E*u_el_B_xx))   +E*(2*u_el_B_xy-u_el_A_xy)) ,('x'))")
problem.add_equation("V_B = 0")






#%% INITIAL CONDITIONS

'FILAMENTS PHASE FIELD'
f_A['g'] = filament_function(f_A['g'], xl_A, xr_A, x[0], li)
f_B['g'] = filament_function(f_B['g'], xl_B, xr_B, x[0], li)
f_D['g'] = np.minimum(f_A['g'],f_B['g'])

'ENERGY LANDSCAPE'
# E_A['g'] = B*(1-f_A['g']) + Eb_A*f_A['g']
# E_B['g'] = B*(1-f_B['g']) + Eb_B*f_B['g']
# E_D['g'] = B*(1-f_D['g']) + Eb_D*f_D['g']

E_A['g'] = -np.log( f_A['g']/(1-f_A['g'] + np.exp(Eb_A)) + np.exp(-B) )
E_B['g'] = -np.log( f_B['g']/(1-f_B['g'] + np.exp(Eb_B)) + np.exp(-B) )
E_D['g'] = -np.log( f_D['g']/(1-f_D['g'] + np.exp(Eb_D)) + np.exp(-B) )

'NUMBER OF PARTICLE'
# 1st guess on the number of particle definition
n_A['g'] = 1*1/( 1+np.exp(E_A['g']) )
n_B['g'] = 1*1/( 1+np.exp(E_B['g']) )
n_D['g'] = 1*1/( 1+np.exp(E_D['g']) )

# n_A['g'] = 0.0*np.exp(-100*(x[0]+1)**2)
# n_B['g'] = 0.0*np.exp(-100*(x[0]-1)**2)
# n_D['g'] = 0.0*np.exp(-100*(x[0]-0.0)**2)


# Particles equilibrium density
# A_eq['g'] = (1-n_D['g'])/(1+np.exp(E_A['g']))
# B_eq['g'] = (1-n_D['g'])/(1+np.exp(E_B['g']))

# D_eq_A['g'] = 0.5* ( 1+n_A['g']*np.exp(E_A['g']-E_D['g']) -np.sqrt( (1+n_A['g']*np.exp(E_A['g']-E_D['g']))**2 -4*n_A['g']*(1-n_B['g'])*np.exp(E_A['g']-E_D['g'])  )  )
# D_eq_B['g'] = 0.5* ( 1+n_B['g']*np.exp(E_B['g']-E_D['g']) -np.sqrt( (1+n_B['g']*np.exp(E_B['g']-E_D['g']))**2 -4*n_B['g']*(1-n_A['g'])*np.exp(E_B['g']-E_D['g'])  )  )



# 'Loop to have the right density'
# for i in range(10000):
#     A_eq['g'] = (1-n_D['g'])/(1+np.exp(E_A['g']))
#     B_eq['g'] = (1-n_D['g'])/(1+np.exp(E_B['g']))

#     D_eq_A['g'] = 0.5* ( 1+n_A['g']*np.exp(E_A['g']-E_D['g']) -np.sqrt( (1+n_A['g']*np.exp(E_A['g']-E_D['g']))**2 -4*n_A['g']*(1-n_B['g'])*np.exp(E_A['g']-E_D['g'])  )  )
#     D_eq_B['g'] = 0.5* ( 1+n_B['g']*np.exp(E_B['g']-E_D['g']) -np.sqrt( (1+n_B['g']*np.exp(E_B['g']-E_D['g']))**2 -4*n_B['g']*(1-n_A['g'])*np.exp(E_B['g']-E_D['g'])  )  )
    
#     n_A['g'] = A_eq['g']
#     n_B['g'] = B_eq['g']
#     n_D['g'] = 0.5*(D_eq_A['g'] + D_eq_B['g']) 
    

A_eq['g'] = (f_A['g']-n_D['g'])/(1+np.exp(Eb_A))
B_eq['g'] = (f_B['g']-n_D['g'])/(1+np.exp(Eb_B))
D_eq_A['g'] = 0.5* ( f_D['g']+n_A['g']*np.exp(Eb_A-Eb_D) -np.sqrt( (f_D['g']+n_A['g']*np.exp(Eb_A-Eb_D))**2 -4*f_A['g']*n_A['g']*(f_B['g']-n_B['g'])*np.exp(Eb_A-Eb_D)  )  )
D_eq_B['g'] = 0.5* ( f_D['g']+n_B['g']*np.exp(Eb_B-Eb_D) -np.sqrt( (f_D['g']+n_B['g']*np.exp(Eb_B-Eb_D))**2 -4*f_B['g']*n_B['g']*(f_A['g']-n_A['g'])*np.exp(Eb_B-Eb_D)  )  )
  

'Loop to have the right density'
for i in range(1000):
    A_eq['g'] = (f_A['g']-n_D['g'])/(1+np.exp(Eb_A))
    B_eq['g'] = (f_B['g']-n_D['g'])/(1+np.exp(Eb_B))

    D_eq_A['g'] = 0.5* ( f_D['g']+n_A['g']*np.exp(Eb_A-Eb_D) -np.sqrt( (f_D['g']+n_A['g']*np.exp(Eb_A-Eb_D))**2 -4*f_A['g']*n_A['g']*(f_B['g']-n_B['g'])*np.exp(Eb_A-Eb_D)  )  )
    D_eq_B['g'] = 0.5* ( f_D['g']+n_B['g']*np.exp(Eb_B-Eb_D) -np.sqrt( (f_D['g']+n_B['g']*np.exp(Eb_B-Eb_D))**2 -4*f_B['g']*n_B['g']*(f_A['g']-n_A['g'])*np.exp(Eb_B-Eb_D)  )  )
    

    n_A['g'] = A_eq['g']
    n_B['g'] = B_eq['g']
    n_D['g'] = 0.5*(D_eq_A['g']+D_eq_B['g'])
       


N_A = integrate.simpson(n_A['g'].reshape(len(n_A['g'])),np.transpose(x[0])[0])
N_B = integrate.simpson(n_B['g'].reshape(len(n_B['g'])),np.transpose(x[0])[0])
N_D = integrate.simpson(n_D['g'].reshape(len(n_D['g'])),np.transpose(x[0])[0])

print('The total occupancy rate of particle are A=%0.4f, B=%0.4f, D=%0.4f'%(N_A,N_B,N_D))
    

#%%
V_A['g'] = 0
V_B['g'] = 0
V_M['g'] = 0

u_el_A_xx['g'] = 0
u_el_A_xy['g'] = 0
u_el_A_yx['g'] = 0
u_el_A_yy['g'] = 0

u_el_B_xx['g'] = 0
u_el_B_xy['g'] = 0
u_el_B_yx['g'] = 0
u_el_B_yy['g'] = 0


FA = scipy.integrate.simpson(  np.transpose(np.array(f_A['g']*(m*0.5*( V_M['g'] - V_A['g'] ))) ), np.transpose(x[0])[0])

F_A['g'] = FA[0]

#%% PLOTTING THE INITIAL CONDITION


'PARTICLES'
plt.figure(dpi=200)
plt.title(label='Initial particles densities')
plt.plot(x[0],n_A['g'],label="A")
plt.plot(x[0],n_B['g'],label="B")
plt.plot(x[0],n_D['g'],label="D")


plt.plot(x[0],f_A['g'],label=r"$f^A$")
plt.plot(x[0],f_B['g'],label=r"$f^B$")
plt.plot(x[0],f_D['g'],label=r"$f^D$")


# plt.plot(x[0],A_eq['g'],label=r"$f^A$")
# plt.plot(x[0],B_eq['g'],label=r"$f^B$")
# plt.plot(x[0],D_eq_A['g'],label=r"$f^D$")
# plt.plot(x[0],D_eq_B['g'],label=r"$f^D$")
# plt.plot(x[0],D_eq['g'],label=r"$f^D$")

plt.legend()
plt.ylim(-0.1,1.1)
plt.show()

# 'ENERGIES'
# plt.figure(dpi=200)
# plt.title(label='Initial energy landscape')
# plt.plot(x[0],E_A['g'],label=r"$E^A$")
# plt.plot(x[0],E_B['g'],label=r"$E^B$")
# plt.plot(x[0],E_D['g'],label=r"$E^D$")
# plt.legend()
# # plt.ylim(-0.1,1.1)
# plt.show()

#%% SETTING THE VELOCITIES FOR THE SIMULATION

tv = np.linspace(0,stop_time,int(stop_time/timestep)+100)
v_A = np.ones(len(tv))
v_B = np.ones(len(tv))

pos_xl_A = np.zeros(len(tv))
pos_xr_A = np.zeros(len(tv))

pos_xl_B = np.zeros(len(tv))
pos_xr_B = np.zeros(len(tv))


# 'Sinusoidale movment'
# f=6
# amp = 0.1
# v_A = 2*np.pi*f*amp*np.cos(2*np.pi*f*tv)
# v_B = -2*np.pi*f*amp*np.cos(2*np.pi*f*tv)

'Pulling steps'
v_A = 0*tv
t_steps = 5e-3
l_steps = 0
N_steps = 2
t0_steps = np.linspace(stop_time/N_steps,stop_time-stop_time/N_steps,N_steps)

# plt.plot(t0_steps)
# for TAU in t0_steps:
    # v_A = v_A + (-1)*func_step(v_A, t_steps, TAU, l_steps, tv)
    
v_A = v_A + (1)*func_step(v_A,2e-2,0.2,0.1,tv)
# v_A = v_A + (1)*func_step(v_A,2e-2,0.4,0.1,tv)
v_A = v_A + (1)*func_step(v_A,2e-2,0.6,0.1,tv)
# v_A = v_A + (1)*func_step(v_A,2e-2,0.8,0.1,tv)

# v_A = v_A + (-1)*func_step(v_A,1e-2,0.25,0.1,tv)
# v_A = v_A + (-1)*func_step(v_A,1e-2,0.35,0.1,tv)
v_B = -v_A
# v_A = 0*v_A





# 'constant velocity'
# Vc = -0
# v_A = Vc*np.ones(len(tv))
# v_B = -Vc*np.ones(len(tv))

'Plotting'
pos_xl_A[0]=xl_A
pos_xr_A[0]=xr_A
pos_xl_B[0]=xl_B
pos_xr_B[0]=xr_B


for i in range(len(tv)-1):
    pos_xl_A[i+1]=pos_xl_A[i]+v_A[i]*timestep
    pos_xr_A[i+1]=pos_xr_A[i]+v_A[i]*timestep
    pos_xl_B[i+1]=pos_xl_B[i]+v_B[i]*timestep
    pos_xr_B[i+1]=pos_xr_B[i]+v_B[i]*timestep
   
plt.figure(dpi=200)  
plt.title(label="Position of filaments with respect to time")  
plt.plot(pos_xl_A,tv,color="blue",label="A")
plt.plot(pos_xr_A,tv,color="blue")
plt.plot(pos_xl_B,tv,color="red",label="B")
plt.plot(pos_xr_B,tv,color="red")
plt.legend()
plt.show()  

plt.figure(dpi=200)  
plt.title(label="velocity of filaments with respect to time")  
plt.plot(v_A,tv,color="blue",label="A")
plt.plot(v_B,tv,color="red",label="B")
plt.legend()
plt.show()  
  
#%% PLOTTING INFORNATION ABOUT VELOCITY, POSTITIONS AND OVERLAP LENGTH

plt.figure(dpi=200)  
plt.title(label="Overlap length") 
plt.plot(tv,pos_xr_A-pos_xl_B,color="black",label="Overlap length")
plt.legend()
plt.show()



# %% BUILDING SOLVER
solver = problem.build_solver(timestepper,ncc_cutoff=1e-4)
solver.stop_sim_time = stop_time



# %% Setting the saving
date = datetime.datetime.now()
name = str(name_save)

analysis = solver.evaluator.add_file_handler(name_save, sim_dt=stop_time/N_save, max_writes=N_save)
# analysis.add_tasks(solver.state, layout='g') # Save all quantities in equations
analysis.add_task(n_A,layout = 'g',name = 'n_A')
analysis.add_task(n_B,layout = 'g',name = 'n_B')
analysis.add_task(n_D,layout = 'g',name = 'n_D')

analysis.add_task(E_A + np.log((n_A+h)/(1-n_A-n_D)),layout = 'g',name = 'MU_A')
analysis.add_task(E_B + np.log((n_B+h)/(1-n_B-n_D)),layout = 'g',name = 'MU_B')
analysis.add_task(E_D + np.log((n_D+h)*(1-n_D)/(1-n_A-n_D)/(1-n_B-n_D)),layout = 'g',name = 'MU_D')

analysis.add_task(f_A,layout = 'g',name = 'f_A')
analysis.add_task(f_B,layout = 'g',name = 'f_B')
analysis.add_task(f_D,layout = 'g',name = 'f_D')

analysis.add_task(E_A,layout = 'g',name = 'E_A')
analysis.add_task(E_B,layout = 'g',name = 'E_B')
analysis.add_task(E_D,layout = 'g',name = 'E_D')

analysis.add_task(A_eq,layout = 'g',name = 'A_eq')
analysis.add_task(B_eq,layout = 'g',name = 'B_eq')
analysis.add_task(D_eq_A,layout = 'g',name = 'D_eq_A')
analysis.add_task(D_eq_B,layout = 'g',name = 'D_eq_B')

analysis.add_task(V_B,layout = 'g',name = 'V_B')
analysis.add_task(V_A,layout = 'g',name = 'V_A')
analysis.add_task(V_M,layout = 'g',name = 'V_M')

analysis.add_task(F_A,layout = 'g',name = 'F_A')
analysis.add_task(F_B,layout = 'g',name = 'F_B')

analysis.add_task(u_el_A_xx,layout = 'g',name = 'u_el_A_xx')
analysis.add_task(u_el_A_xy,layout = 'g',name = 'u_el_A_xy')
analysis.add_task(u_el_A_yx,layout = 'g',name = 'u_el_A_yx')
analysis.add_task(u_el_A_yy,layout = 'g',name = 'u_el_A_yy')

analysis.add_task(u_el_B_xx,layout = 'g',name = 'u_el_B_xx')
analysis.add_task(u_el_B_xy,layout = 'g',name = 'u_el_B_xy')
analysis.add_task(u_el_B_yx,layout = 'g',name = 'u_el_B_yx')
analysis.add_task(u_el_B_yy,layout = 'g',name = 'u_el_B_yy')

analysis.add_task(f_ent,layout = 'g',name = 'f_ent')

analysis.add_task(force_B,layout = 'g',name = 'force_B')
analysis.add_task(force_elastic_B,layout = 'g',name = 'force_elastic_B')
analysis.add_task(force_viscous_B,layout = 'g',name = 'force_viscous_B')



# %% Starting the main loop
print("Start")
j=0
t=0
T_N0 = datetime.datetime.now()
while solver.proceed:
    f_D['g'] = np.minimum(f_A['g'],f_B['g'])

    # f_B.change_scales(1)
    # V_B.change_scales(1)
    # V_M.change_scales(1)
    
    # FB = scipy.integrate.simpson(  -1*np.transpose(np.array(f_B['g']*(m*0.5*( V_B['g'] - V_M['g'] ))) ), np.transpose(x[0])[0])
    # F_B['g'] = FB[0]
    solver.step(timestep) # solving the equations
    # V_A.change_scales(dealias)
    # V_B.change_scales(dealias)
    
    V_A['g'] = v_A[t]
    V_B['g'] = v_B[t]
    t=t+1        

    
    n_A['g'][n_A['g']<0] = 0
    n_B['g'][n_B['g']<0] = 0
    n_D['g'][n_D['g']<0] = 0

    
    if solver.iteration % int(stop_time/(N_save*timestep)) == 0 :
    # if solver.iteration % 1 == 0 :

        j=j+1
        T_N1 = datetime.datetime.now()
        T_LEFT = (T_N1-T_N0)*(N_save-j)
        logger.info('%i/%i, T=%0.2e, t_left = %s' %(j,N_save,solver.sim_time,str(T_LEFT)))
        T_N0 = datetime.datetime.now()
  
        if j%10  == 0:
                
                # V_A.change_scales(1)
                # V_B.change_scales(1)
                # V_M.change_scales(1)
                # # # f_ent.change_scales(1)
                                
                # plt.plot(x[0],V_A['g']-1,color = 'blue',label = "V_A")
                # plt.hlines(-1, -3, 3, color = 'blue', alpha = 0.5)
                
                # plt.plot(x[0],V_B['g']+1,color = 'red',label = "V_B")
                # plt.hlines(1, -3, 3, color = 'red', alpha = 0.5)

                # plt.plot(x[0],V_M['g'],color = 'black',label = "V_M")
                # plt.hlines(0, -3, 3, color = 'black', alpha = 0.5)
                
                # # plt.plot(x[0],Ly/2*f_ent['g'])
                # plt.show()
                
                # U_A.change_scales(1)
                # plt.plot(x[0],U_A['g'],color = 'black',label = "U_A")
                # plt.show()


                n_A.change_scales(1)
                n_B.change_scales(1)
                n_D.change_scales(1)
                
                plt.plot(x[0],n_A['g']-1,color = 'blue',label = "n_A")
                plt.hlines(-1, -3, 3, color = 'blue', alpha = 0.5)
                
                plt.plot(x[0],n_B['g']+1,color = 'red',label = "n_B")
                plt.hlines(1, -3, 3, color = 'red', alpha = 0.5)

                plt.plot(x[0],n_D['g'],color = 'black',label = "n_D")
                plt.hlines(0, -3, 3, color = 'black', alpha = 0.5)
                plt.show()
                
                # u_el_A_xy.change_scales(1)
                # u_el_B_xy.change_scales(1)
                                
                # plt.plot(x[0],u_el_A_xy['g']-1,color = 'blue',label = "u_el_A")
                # plt.hlines(-1, -3, 3, color = 'blue', alpha = 0.5)
                
                # plt.plot(x[0],u_el_B_xy['g']+1,color = 'red',label = "u_el_B")
                # plt.hlines(1, -3, 3, color = 'red', alpha = 0.5)
                # plt.legend()
                # plt.show()
                
                print(V_B['g'][0][0],force_B['g'][0][0])



#%%
#%%
 

#%%

#%%




# %% Getting the saved files#
tasks = d3.load_tasks_to_xarray(name_save +"/"+name_save_file+"_s1.h5") # Downloadig the files


#%%
n_img = 10
N_end = N_save
# %% ANIMATION
fig = plt.figure(dpi=200)
def animate(i):
    if i%(N_save/100) == 0:
        print(i)
    plt.clf()
    plt.plot(x[0],tasks['n_A'][i],color = 'blue',label = "Particles A")
    plt.plot(x[0],tasks['A_eq'][i],color = 'blue',alpha=0.5)

    plt.plot(x[0],tasks['n_B'][i],color = 'red',label = "Particles B")
    plt.plot(x[0],tasks['B_eq'][i],color = 'red',alpha=0.5)

    plt.plot(x[0],tasks['n_D'][i],color = 'purple',label = "Particles D")
    plt.plot(x[0],tasks['D_eq_A'][i],color = 'purple',alpha=0.5)
    plt.plot(x[0],tasks['D_eq_B'][i],color = 'purple',alpha=0.5)
    
    plt.plot(x[0],tasks['f_A'][i], color='black')
    plt.plot(x[0],tasks['f_B'][i], color='black')

    plt.hlines(1,-Lx,+Lx)    
    # plt.legend()
    plt.ylim(-0.1,1.1)
    plt.xlim(-Lx/2, Lx/2)
    
t=np.arange(0,N_end,n_img) # New time array with only n images  
ani = FuncAnimation(fig, animate, frames=t,
                    interval=1, repeat=False)
#name = "D"+str(D)+"a"+str(alpha)+".gif"
ani.save(name_save+"_density"+extension, writer = 'ffmpeg', fps = 20)

# %% ANIMATION
fig = plt.figure(dpi=200)
def animate(i):
    if i%(N_save/100) == 0:
        print(i)
    plt.clf()
    plt.plot(x[0],tasks['f_A'][i]*tasks['MU_A'][i],color = 'blue',label = "Particles A")
    # plt.plot(x[0],tasks['n_A'][i]*tasks['MU_A'][i],color = 'blue',label = "f A",linestyle = '--')

    # plt.plot(x[0],tasks['E_A'][i],color = 'blue',alpha = 0.5)

    plt.plot(x[0],tasks['f_B'][i]*tasks['MU_B'][i],color = 'red',label = "Particles B")
    # plt.plot(x[0],tasks['n_B'][i]*tasks['MU_B'][i],color = 'red',label = "f B",linestyle = '--')

    # plt.plot(x[0],tasks['E_B'][i],color = 'red',alpha = 0.5)

    plt.plot(x[0],tasks['f_D'][i]*tasks['MU_D'][i],color = 'purple',label = "Particles D")
    # plt.plot(x[0],tasks['n_D'][i]*tasks['MU_D'][i],color = 'purple',label = "f D",linestyle = '--')

    # plt.plot(x[0],tasks['E_D'][i],color = 'purple',alpha = 0.5)

    # plt.hlines(1,-Lx,+Lx)    
    # plt.legend()
    # plt.ylim(-0.1,1.1)
    plt.xlim(-Lx/2, Lx/2)
    plt.legend()
    
t=np.arange(0,N_end,n_img) # New time array with only n images  
ani = FuncAnimation(fig, animate, frames=t,
                    interval=1, repeat=False)
#name = "D"+str(D)+"a"+str(alpha)+".gif"
ani.save(name_save+"_chemical_potential"+extension, writer = 'ffmpeg', fps = 20)


#%%
# %% ANIMATION
fig = plt.figure(dpi=200)
def animate(i):
    if i%(N_save/100) == 0:
        print(i)
        plt.clf()
        plt.plot(x[0],(tasks['V_A'][i]*tasks['f_A'][i])-1,color = 'blue',label = "V_A")
        plt.hlines(-1, -3, 3, color = 'blue', alpha = 0.5)
        
        plt.plot(x[0],(tasks['V_B'][i]*tasks['f_B'][i])+1,color = 'red',label = "V_B")
        plt.hlines(1, -3, 3, color = 'red', alpha = 0.5)
        
        plt.plot(x[0],tasks['V_M'][i],color = 'black',label = "V_M")
        plt.hlines(0, -3, 3, color = 'black', alpha = 0.5)

    # plt.legend()
    # plt.ylim(-0.1,1.1)
    plt.xlim(-Lx/2, Lx/2)
    
t=np.arange(0,N_end,n_img) # New time array with only n images  
ani = FuncAnimation(fig, animate, frames=t,
                    interval=1, repeat=False)
#name = "D"+str(D)+"a"+str(alpha)+".gif"
ani.save(name_save+"_V"+extension, writer = 'ffmpeg', fps = 20)

# %% ANIMATION
fig = plt.figure(dpi=200)
def animate(i):
    if i%(N_save/100) == 0:
        print(i)
        plt.clf()
        plt.hlines(-1, -3, 3, color = 'blue', alpha = 0.5)
        plt.hlines(1, -3, 3, color = 'red', alpha = 0.5)
        plt.plot(x[0],tasks['u_el_A_xy'][i]-1,color = 'blue',label = "u_el_A")       
        plt.plot(x[0],tasks['u_el_B_xy'][i]+1,color = 'red',label = "u_el_B")
        
    # plt.legend()
    # plt.ylim(-0.1,1.1)
    plt.xlim(-Lx/2, Lx/2)
    
t=np.arange(0,N_end,n_img) # New time array with only n images  
ani = FuncAnimation(fig, animate, frames=t,
                    interval=1, repeat=False)
#name = "D"+str(D)+"a"+str(alpha)+".gif"
ani.save(name_save+"_u_el"+extension, writer = 'ffmpeg', fps = 20)

# %% ANIMATION
fig = plt.figure(dpi=200)
def animate(i):
    if i%(N_save/100) == 0:
        print(i)
        plt.clf()
        plt.plot(x[0],tasks['f_B'][i]*(2*tasks['u_el_B_xy'][i]-tasks['u_el_A_xy'][i]),label = "F_el")
        
    # plt.legend()
    # plt.ylim(-0.1,1.1)
    plt.xlim(-Lx/2, Lx/2)
    
t=np.arange(0,N_end,n_img) # New time array with only n images  
ani = FuncAnimation(fig, animate, frames=t,
                    interval=1, repeat=False)
#name = "D"+str(D)+"a"+str(alpha)+".gif"
ani.save(name_save+"_F_u_el"+extension, writer = 'ffmpeg', fps = 20)



#%%

fig = plt.figure(dpi=200)
def animate(i):
    if i%(N_save/100) == 0:
        print(i)
        plt.clf()
        plt.plot(x[0],tasks['u_el_B_xx'][i]-1.5,label = "xx")       
        plt.plot(x[0],tasks['u_el_B_xy'][i]-0.5,label = "xy")       
        plt.plot(x[0],tasks['u_el_B_yx'][i]+0.5,label = "yx")       
        plt.plot(x[0],tasks['u_el_B_yy'][i]+1.5,label = "yy")       

    plt.legend()
    # plt.ylim(-0.1,1.1)
    plt.xlim(-Lx/2, Lx/2)
    
t=np.arange(0,N_end,n_img) # New time array with only n images  
ani = FuncAnimation(fig, animate, frames=t,
                    interval=1, repeat=False)
#name = "D"+str(D)+"a"+str(alpha)+".gif"
ani.save(name_save+"_u_el_B"+extension, writer = 'ffmpeg', fps = 20)

#%% FORCE

t = np.linspace(0, stop_time,len(tasks['F_A']))
Force_B = np.zeros(len(t))
Force_B_el = np.zeros(len(t))
Force_B_visc = np.zeros(len(t))

for i in range(len(t)):
    Force_B[i] = tasks['force_B'][i][20] 
    Force_B_el[i] = tasks['force_elastic_B'][i][20] 
    Force_B_visc[i] = tasks['force_viscous_B'][i][20] 


#%%
plt.figure(dpi = 300)
# plt.plot(t,Force_A,color = 'blue',label = "F_A")
plt.plot(t,Force_B,label="tot")
# plt.plot(t,Force_B_el,label="el")
# plt.plot(t,Force_B_visc,label="visc")

plt.legend()
plt.show()


#%%
t = np.linspace(0, stop_time,len(tasks['V_B']))
VB = np.zeros(len(t))


for i in range(len(t)):
    VB[i] = tasks['V_B'][i][20] 
    
#%%
plt.figure(dpi = 300)
# plt.plot(t,Force_A,color = 'blue',label = "F_A")
plt.plot(t,VB,label="VB")


plt.legend()
plt.show()

#%%

'FIN'




#%% chemical potential analyse


#%%
#%%

#%%




