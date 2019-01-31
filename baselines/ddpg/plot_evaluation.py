import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = "eval.dat"

figures = np.array(pd.read_csv(filepath_or_buffer=filename, sep=" ", header=None))

print(figures[44:544].T[0])

figures_x = figures[44:544].T[0]
figures_y = figures[44:544].T[1]
figures_y_2 = figures[544:1044].T[1]
avg = (figures_y + figures_y_2) / 2

# avg = np.mean(figures, axis=0)[1]
# maximum = np.mean(np.max(figures, axis=2).T[1])
# print(maximum)
# index = (np.argmax(figures, axis=2).T[1])
# avg_error = 0
# for i in range(5):
#     avg_error += figures[i][2][index[i]]
# avg_error = avg_error / 5
# print(avg_error)
#
plt.subplot(3, 1, 1)
plt.plot(figures_x, avg, label="hopper_avg")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.subplot(3, 1, 2)
plt.plot(figures_x, figures_y, label="hopper_1")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.subplot(3, 1, 3)
plt.plot(figures_x, figures_y_2, label="hopper_2")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()
# plt.savefig("Hopper.png")
