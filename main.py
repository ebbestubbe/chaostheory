# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:02:42 2017

@author: Ebbe
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

# Plot a scatter that persists (isn't redrawn) and the initial line.
x = np.arange(0, 20, 0.1)
ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
line, = ax.plot(x, x - 5, 'r-', linewidth=2)

def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    line.set_ydata(x - 5 + i)
    ax.set_xlabel(label)
    return line, ax

if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    #if len(sys.argv) > 1 and sys.argv[1] == 'save':
    #anim.save('line.gif', dpi=80, writer='imagemagick')
    anim.save('basic_animation.mp4', fps=30)
    plt.show()
    #else:
        # plt.show() will just loop the animation forever.
    #    


'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation
import math

p0 = [0,0]
#L = math.sqrt((p1[0]-p0[0])**2 + (p1[1] - p0[1])**2)
L = 10
g = 9.82

def deriv_z(var,t): #var:[theta,d(theta)/dt]
    
    dy0 = var[1]
    dy1 = -g/L*math.sin(var[0])
    return [dy0,dy1]

t = np.arange(0,5.0,0.05)
init = [1,0]
z = integrate.odeint(deriv_z,init,t)
x = -L*np.sin(z[:,0])
y = -L*np.cos(z[:,0])
plt.close("all")
plt.figure()
plt.plot(t,x)


fig, ax = plt.subplots()
fig.set_tight_layout(True)
dot, = ax.plot(x[0], y[0], 'bo')
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))
def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    
    dot.set_xdata(x[i+1])
    dot.set_ydata(y[i+1])
    ax.set_xlabel(label)
    return dot, ax

# FuncAnimation will call the 'update' function for each frame; here
# animating over 10 frames, with an interval of 200ms between frames.
anim = FuncAnimation(fig, update, frames=np.arange(0,len(t)), interval=2)
#anim.save('line.gif', dpi=80, writer='imagemagick')
plt.show()
'''
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