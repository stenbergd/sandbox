#!/usr/bin/python3

'''
SIR Model

S, Susceptible
I, Infectious
R, Recovered

dS(t)/dt = -Beta * I(t) * S(t)
dI(t)/dt = Beta * I(t) * S(t) - Gamma * I(t)
dR(t)/dt = Gamma * I(t)

S(t) + I(t) + R(t) = 1

dS(t)/dt + dI(t)/dt + dR(t)/dt = 0

R0 = Beta / Gamma, basic reproduction number
dI(t)/dt > 0 <---> R0 > 1 ---> epidemic

R0 = 12-18 for measles with 7-10 days of infectious period
R0 = 1.4-3.9 for covid-19 with ~14 days infectious period

Derivates in above differential equations discretized using backward Eulers method
Sample time must be small enough to provide stable results with sufficient accuracy
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def model_func(y, t, beta, gamma):
	f = [ -beta * y[1] * y[0], beta * y[1] * y[0] - gamma * y[1], gamma * y[1] ]
	return f

parser = argparse.ArgumentParser(description='SIR Model epidemic simulator')

parser.add_argument('--beta', '-b', action="store", dest="beta", type=float, default=2.3/14.0)
parser.add_argument('--gamma', '-g', action="store", dest="gamma", type=float, default=1/14.0)
parser.add_argument('--days', '-d', action="store", dest="days", type=int, default=180)

args = parser.parse_args()

beta = args.beta
gamma = args.gamma
days = args.days

h = 0.1 # sample time
t = np.linspace(0, days, (days+1)/h)

r0 = beta/gamma

y0 = [ 0.999, 0.001, 0.0 ] # Initial susceptible, infectious and recovered population ratios

y = odeint(model_func, y0, t, args=(beta, gamma))

plt.plot(t, y[:, 0], 'b', label='S(t) - Susceptible')
plt.plot(t, y[:, 1], 'r', label='I(t) - Infectious')
plt.plot(t, y[:, 2], 'g', label='R(t) - Recovered')
plt.legend()
plt.title('SIR Model - R0: {} (Beta: {}, Gamma: {})'.format(round(r0, 2), round(beta, 2), round(gamma, 2)))
plt.xlabel('Time [days]')
plt.ylabel('Population ratio')
plt.show()