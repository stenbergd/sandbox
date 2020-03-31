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

import matplotlib.pyplot as plt
import numpy as np

days = 180
h = 0.1 # sample time
t = np.linspace(0, days, (days+1)/h)

Gamma = 1/14.0
Beta = 2.3*Gamma

s = np.zeros(int((days+1)/h))
i = np.zeros(int((days+1)/h))
r = np.zeros(int((days+1)/h))

s[0] = 0.99 # Initial susceptible population ratio
i[0] = 0.01 # Initial infectious population ratio
r[0] = 0.0 # Initial recovered population ratio

for k in range(0, len(t)-1):
	s[k+1] = s[k] - Beta*i[k]*s[k]*h
	i[k+1] = i[k] + Beta*i[k]*s[k]*h - Gamma*i[k]*h
	r[k+1] = r[k] + Gamma*i[k]*h

plt.plot(t, s, 'b', label='S(t) - Susceptible')
plt.plot(t, i, 'r', label='I(t) - Infectious')
plt.plot(t, r, 'g', label='R(t) - Recovered')
plt.legend()
plt.title('SIR Model - R0: {} (Beta: {}, Gamma: {})'.format(round(Beta/Gamma, 2), round(Beta, 2), round(Gamma, 2)))
plt.xlabel('Time [days]')
plt.ylabel('Population ratio')
plt.show()