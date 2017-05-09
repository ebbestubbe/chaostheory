# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:02:42 2017

@author: Ebbe
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
plt.close()
p0 = [0,0]
#L = math.sqrt((p1[0]-p0[0])**2 + (p1[1] - p0[1])**2)
L = 10
g = 9.82

def deriv_z(var,t): #var:[theta,d(theta)/dt]
    
    dy0 = var[1]
    dy1 = -g/L*math.sin(var[0])
    return [dy0,dy1]

t = np.arange(0,25.0,0.1)
init = [1,0]
z = integrate.odeint(deriv_z,init,t)
plt.figure()
plt.plot(t,z[:,0])
plt.figure()
plt.plot(t,z[:,1])
x = -L*np.sin(z[:,0])
y = -L*np.cos(z[:,0])
plt.figure()
plt.plot(x,y)
plt.axis([-1.1*L,1.1*L,-1.1*L,1.1*L])
plt.show()

'''

def deriv(var,t):
    x1,vx1,y1,vy1 = var
    #theta measured from vertical line to string in positive rotation
    costheta = (y1-p0[1])/L
    sintheta = (x1-p0[0])/L
    ax1 = -g*(costheta)
    ay1 = g*(sintheta)
    return [vx1,ax1,vy1,ay1]
        
t = np.arange(0, 5.0, 0.01)
init = [p1[0],0,p1[1],0]
z = integrate.odeint(deriv,init,t)
plt.figure()
plt.plot(t,z[:,0])
plt.show()
plt.figure()
plt.plot(t,z[:,2])
plt.show()


plt.figure()
plt.plot(z[:,0],z[:,2])
#print(z)
'''
'''
pi = np.pi
sqrt = np.sqrt
cos = np.cos
sin = np.sin

def deriv_z(z, phi):
    u, udot = z
    return [udot, -u + sqrt(u)]

phi = np.linspace(0, 7.0*pi, 2000)
zinit = [1.49907, 0]
z = integrate.odeint(deriv_z, zinit, phi)
u, udot = z.T
# plt.plot(phi, u)
fig, ax = plt.subplots()
ax.plot(1/u*cos(phi), 1/u*sin(phi))
ax.set_aspect('equal')
plt.grid(True)
plt.show()
'''