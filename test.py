import random
import pandas as pd
import matplotlib.pyplot as plt
all_trials = {}
for j in range(30):
    nr_history = []
    nr_list = [str(random.randint(0, 9)) for i in range(4)]

    while True:
        nr = "".join(nr_list)
        print(nr)
        nr_history.append(int(nr))
        if nr == "6174" or nr == '0':
            print('\n\n\n')
            break
        nr_low = ''.join(sorted(nr_list))
        nr_high = ''.join(sorted(nr_list, reverse=True))
        nr_new = int(nr_high) - int(nr_low)
        nr_list = [char for char in str(nr_new)]

    all_trials[str(j)] = nr_history

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_trials.items()]))
# df = pd.DataFrame(all_trials)
df.fillna(6174, inplace=True)
df.plot(kind='line')
plt.grid()
plt.show()