# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:02:42 2017

@author: Ebbe

"""

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.01
t = np.arange(0.0, 10, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = -10.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])
#color = '0.2'
color = [0.3,0.3,0.3]
# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

#fig = plt.figure(figsize=(15,15))
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
plt.gca().set_aspect('equal', adjustable='box')

line, = ax.plot([], [], 'o-',color=color, lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#sub1 = fig.add_subplot(1,1,1)
#sub1.xaxis.set_visible(False)
#sub1.yaxis.set_visible(False)
#sub1.set_xlim([-2, 2])
#sub1.set_ylim([-2, 2])
#sub1.axis('off')

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

ani.save('double_pendulum.mp4', fps=15)

fig2 = plt.figure()
ax1 = fig2.add_subplot(221)
ax1.plot(t,y[:,0],color=color)
ax2 = fig2.add_subplot(222)
ax2.plot(t,y[:,1],color=color)
ax3 = fig2.add_subplot(223)
ax3.plot(t,y[:,2],color=color)
ax4 = fig2.add_subplot(224)
ax4.plot(t,y[:,3],color=color)

print("done")
plt.show()

'''
import matplotlib
matplotlib.use("Agg") # configure backend for PNGs
import matplotlib.pyplot as plt # make figures
from matplotlib import patches # polygon shapes
import math # square roots and other calculations
import imageio # save images
import scipy # rotation calculations
import random 
import os # file deletion
import scipy.integrate as integrate
import numpy as np

def main():
    fig = plt.figure(figsize=(30, 30))
    plt.subplots_adjust(hspace=0, wspace=0)
    
    
    #L = math.sqrt((p1[0]-p0[0])**2 + (p1[1] - p0[1])**2)
    L = 1
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
    n_images = len(t)
    xdata = []
    ydata = []
    for i in range(n_images):
        xdata.append([0,x[i]])
        ydata.append([0,y[i]])
    generateimages(fig,xdata,ydata)
    
    
    images = [] #  Turn a list of images into a GIF using ImageIO
    tag = "experiment2"
    for n in range(n_images):
        readname = str('temp' + repr(n) + '.png')
        images.append(imageio.imread(readname))
    
    savegif(images,tag)    
def generateimages(fig,xdata,ydata):
    
    for i in range(len(xdata)): # for every frame in the GIF
        print("generating",i)
        sub1 = fig.add_subplot(1,1,1)
        sub1.xaxis.set_visible(False)
        sub1.yaxis.set_visible(False)
        sub1.set_xlim([-2, 2])
        sub1.set_ylim([-2, 2])
        sub1.axis('off')
        line, = sub1.plot(xdata[i],ydata[i], '.-', lw=1)
        
        savename = str('temp' + repr(i) + '.png')
        fig.savefig(savename, bbox_inches='tight', pad_inches=0, dpi=50)
        plt.clf() 

def savegif(images,tag):
    
    imageio.mimsave(str(tag) + '.gif', images,fps=60)
    print("saved")
    plt.close('all')
if __name__ == "__main__":
    print("start")
    main()
'''
'''
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