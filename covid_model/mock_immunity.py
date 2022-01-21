import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 365*4, 365*4)
a = 400
b = 400
c = 0.89

x = c * np.exp(-t/a)
if a != b:
    y = (b * np.exp(-(t * (a + b))/(a * b)) * (np.exp(t/b) - np.exp(t/a)))/(a - b)
    z = (c * np.exp(-(t * (a + b))/(a*b)) * (b*np.exp(t/a) - np.exp(t/b) * (b*np.exp(t/a) + a*(-np.exp(t/a)) + a)))/(a - b)
else:
    y = c * t * np.exp(-t/a) / a
    z = (c * np.exp(-t/a) * (a * np.exp(t/a) - a - t)) / a

plt.plot(t, x, label='Recovered 1')
plt.plot(t, y, label='Recovered 2')
plt.plot(t, x+y, label='Recovered Total')
plt.plot(t, z, label='Immunity Lapsed')
plt.legend()
plt.grid()

plt.show()