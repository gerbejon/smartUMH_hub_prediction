import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import pandas as pd

rand = np.random.default_rng(42)
# t =  rand.random(100)
t =  rand.random(100) * 100
# t = np.arange(0,100)
# y = np.sin(2 * np.pi * t) + 0.1 * rand.standard_normal(100)
y = np.sin(t) + 0.1 * rand.standard_normal(100)

frequency, power = LombScargle(t, y).autopower()

plt.plot(frequency, power)
plt.show()

df = pd.DataFrame({'t': t, 'y': y})
df.sort_values('t', inplace=True)
plt.plot(df['t'], df['y'])
x = np.arange(round(t.min()), round(t.max()))

plt.plot(
    x,
    np.sin(x)
         )

plt.show()


from plots.periodicity import _plot_lombscargle
ls = _plot_lombscargle(t, y)
print(ls.sort_values('power', ascending=False))