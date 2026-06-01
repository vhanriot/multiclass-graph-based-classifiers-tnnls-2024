from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def hk(x, max_d):
    return np.exp(max_d / x)


def calc_perc_01_05(max_d):
    all = integrate.quad(hk, max_d / 10, max_d, args=(max_d))[0]
    return integrate.quad(hk, max_d / 10, 2 * max_d / 10, args=(max_d))[0] / all


def calc_perc(max_d, beg, end):
    all = integrate.quad(hk, max_d / 10, max_d, args=(max_d))[0]
    return integrate.quad(hk, beg * max_d, end * max_d, args=(max_d))[0] / all


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "act_fun_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)


max_d = 1
I = integrate.quad(hk, 0.1, 1, args=(max_d))

max_d = 1

if max_d < 1:
    beg = (max_d**2) / 100
else:
    beg = 0.1

x = np.linspace(beg, max_d, 10000)
y = np.exp(max_d**2 / x)

plt.plot(x, y, linewidth=3, c="black")
plt.ylim([0, 30])
x = np.linspace(beg, max_d, 10000)
y = np.exp(max_d**2 / x)
fig, ax = plt.subplots(figsize=(10, 10))
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.plot(x, y, linewidth=5, c="black")
plt.ylim([0, 30])
plt.xlim([0, 1])
plt.ylabel(r"$h_k$", fontsize=25)
plt.xlabel(r"$||\mathbf{x}-\mathbf{p}_k||$", fontsize=25)
x_ticks_labels = [r".1$m_d$", r".25$m_d$", r".4$m_d$", r".55$m_d$", r".7$m_d$", r".85$m_d$", r"$m_d$"]
y_ticks_labels = ["0", r"10$m_d$", r"20$m_d$", r"30$m_d$"]
x_ticks = np.round(np.linspace(0.1, 1, 7), 2)
y_values = np.exp(1 / x_ticks)
ax.set_xticks(x_ticks.tolist())
ax.set_yticks([0, 10, 20, 30])
ax.set_xticklabels(x_ticks_labels, fontsize=15)
ax.set_yticklabels(y_ticks_labels, fontsize=15)

plt.vlines(x_ticks, 0, y_values, colors="black", linestyles="dashed")

midpoints = np.linspace(0.1, 1, 7)[:-1] + 0.15 / 2
y_mid = np.exp(1 / midpoints)
y_plot = y_mid / 2
y_plot[0] = 20
y_plot[1] = 10

areas = [calc_perc(1, 0.1, 0.25)]
areas.append(calc_perc(1, 0.25, 0.4))
areas.append(calc_perc(1, 0.4, 0.55))
areas.append(calc_perc(1, 0.55, 0.7))
areas.append(calc_perc(1, 0.7, 0.85))
areas.append(calc_perc(1, 0.85, 1))


for i in range(len(midpoints)):
    if i == 0:
        minuss = 0.065
    elif i == 1:
        minuss = 0.04
    else:
        minuss = 0.05
    plt.text(midpoints[i] - minuss, y_plot[i], f"{np.round(100 * areas[i], 2)}%", fontsize=20)

plt.savefig(FIG_DIR / "act_fun_expmax.eps", format="eps")
