# A pyhton version for the area_fraction programm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# 21.1.21 @mhansinger

####################################
# Some constants

# length of the cube
dx = 1.0

# limits for the variables: dl, phi, theta
dl_hi = 1.7     # Das Maximum ist sqrt(3), die Diagonale des Würfels
dl_low = 0.1

ndl = 17        # number of dl increments
dl_step=(dl_hi - dl_low)/(ndl-1)

phi_hi = 0
phi_low=90
nphi= 401
phi_step=(phi_hi - phi_low) /(nphi-1)

theta_hi = 90
theta_low = 0
ntheta = 401
theta_step=(theta_hi- theta_low)/(ntheta-1)


#####################################
# now loop over all the different angles and dl values

# loop1
for k in range(0,ndl):
    dl = dl_low + k*dl_step

    #loop 2
    for i in range(0,nphi):
        phi = phi_low + i*phi_step

        #loop 3
        for j in range(0,ntheta):
            theta = theta_low + j*theta_step

            # TODO: function welche die Area fraction berechnet!




def area_fraction(phi,theta,dl):
    '''

    :param phi:
    :param theta:
    :param dl:
    :return: should return "area" and "npoints"
    '''

    #TODO:
    # 1. find the plane-cube intersection points (3,4,5, or 6 but not really needed)
    # 2. get the list of vertices (do I need that?)
    # 3. compute intersection area form the points


    #! for the plane-cube intersection points
    # ! npoints is the number of intersections with the unit cube
    # ! x_intersect, y_intersect, z_intersect store the coordinates of the intersection points
    x_intersect = np.zeros(6)
    y_intersect = np.zeros(6)
    z_intersect = np.zeros(6)

    #! coordinates of the unicube
    vx = np.array([0,1,0,0,1,1,0,1])
    vy = np.array([0,0,1,0,0,1,1,1])
    vz = np.array([0,0,0,1,1,0,1,1])

    #initialize
    npoints = 0
    area = 0

    # check if there is an intersection at all
    dl_max = (np.cos(np.deg2rad(theta)) + np.sin(np.deg2rad(theta))) * np.sin(np.deg2rad(phi)) + np.cos(np.deg2rad(phi))
    if dl > dl_max:
        # exit the function
        return area, npoints

    #! coordinates of the normal vector of length dl
    xn = dl * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    yn = dl * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    zn = dl * np.cos(np.deg2rad(phi))

    #! hessian normal form of the cutting plane, a*x + b*y + c*z = d
    mag = np.sqrt(xn*xn + yn*yn + zn*zn)
    a = xn/mag
    b = yn/mag
    c = zn/mag
    d = dl

