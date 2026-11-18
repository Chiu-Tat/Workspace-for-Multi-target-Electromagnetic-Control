import matplotlib.pyplot as plt
# Run from the repository root: python -m analysis.box_ratio
import numpy as np

size = [0.125, 0.125, 0.3242, 0.3242, 0.4687, 0.4687, 0.5598, 0.6117, 0.6117]
time = [2.35, 2.38, 4.6, 4.2, 10.06, 11.36, 32.1, 104.07, 108.18]

# Handle length mismatch by using the length of the 'size' array as reference
# assuming corresponding data points.
n = min(len(size), len(time))
size_plot = size[:n]
time_plot = time[:n]
x = range(1, n + 1)

fig, ax1 = plt.subplots()

# Plot Size (Bar) on the first y-axis
color = 'tab:blue'
ax1.set_xlabel('Resolution ratio', fontsize=20)
ax1.set_ylabel('Ratio of feasible area', color=color, fontsize=20)
bars = ax1.bar(x, size_plot, color=color, alpha=0.6, label='Size')
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelcolor=color, labelsize=20)

ticksx = [1, 2, 3, 4, 5, 6, 7, 8, 9]
labelsx = [r'$\frac{1}{10}$', r'$\frac{1}{15}$', r'$\frac{1}{20}$', r'$\frac{1}{30}$', r'$\frac{1}{40}$', r'$\frac{1}{60}$', r'$\frac{1}{90}$', r'$\frac{1}{120}$', r'$\frac{1}{180}$']

ax1.set_xticks(ticksx, labelsx)

# Add data labels for Size (Bar) - displayed as percentage
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height * 100:.1f}%',
             ha='center', va='bottom')

# Create second y-axis for Time (Line)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Time (s)', color=color, fontsize=20)  # we already handled the x-label with ax1
ax2.plot(x, time_plot, color=color, marker='o', linestyle='-', linewidth=2, label='Time')
ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

# Add data labels for Time (Line) - displayed as value + 's'
for i, txt in enumerate(time_plot):
    ax2.text(x[i], txt, f'{txt}s', ha='center', va='top')

#plt.title('Size and Time per Data Point')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
