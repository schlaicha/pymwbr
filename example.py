#!/usr/bin/env python

"""
Values based on Kofke Antoine law + Johnson MWBR EOS, corrected for TS
potential (Schlaich & Coasne, JCP 2019)
 - should be correct within 1-2% (based on rigorous finite size scaling,
to be published)
"""

import mwbr

# parameters for Argon
temp = 95.84 # temperature in Kelin
epsilon = 119.8 # LJ epsilon in k_B x Kelvin
sigma = 0.3405 # LJ sigma in nanometer
mass = 39.948

# initialize the MWBR for different cutoffs (measured in sigmas)
eos = {  3:  mwbr.MBWR(rcut=3., sigma = sigma, epsilon = epsilon, mass = mass),
         4:  mwbr.MBWR(rcut=4., sigma = sigma, epsilon = epsilon, mass = mass),
         5:  mwbr.MBWR(rcut=5., sigma = sigma, epsilon = epsilon, mass = mass),
       None: mwbr.MBWR(sigma = sigma, epsilon = epsilon, mass = mass)}

print('# rcut [sigma]\t Tc [Kelvin]\t fsat [bar]')
for rcut,myeos in eos.items():
    Tc = (myeos.critical()[0] / myeos.to_lj_T) [0] # critical temperature in K
    fsat = myeos.fsat(temp*myeos.to_lj_T) / myeos.to_lj_p
    print (rcut, '\t', Tc, '\t', fsat)
