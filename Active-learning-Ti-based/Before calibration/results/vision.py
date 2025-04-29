import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns  # Import seaborn for the color palette

# Define the color palette from the image
palette = ['#EF767A', '#456990', '#48C0AA']

fs = 30
# Set plot parameters
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['font.size'] = fs

# Metrics to plot
metrics = ['r2', 'mis_area', 'oc']
y_labels = ['R$^2$ Score', 'Misclassification Area', 'OC']

# Loop through each metric to create separate figures
for idx, metric in enumerate(metrics):
    fig, axes = plt.subplots(figsize=(8, 6), dpi=300)

    # Loop through methods and plot results for each metric
    for i, method in enumerate(['SVR', 'XGBOOST', 'NN']):
        result = pd.read_csv('Creep rupture life (h)_' + method + '_EI_1' + '.csv')
        x = result['iteration']
        y = result[metric]

        # Plot using the specified color palette
        axes.plot(x, y, '-', alpha=1, zorder=1, label=method, color=palette[i], linewidth=4)

    # Set labels
    axes.set_xlabel('Iteration')
    axes.set_ylabel(y_labels[idx])

    # Customize ticks and spines
    plt.legend(loc="best", frameon=False, fontsize=0.8 * fs)
    plt.tick_params(which='major', length=7, width=2, labelsize=0.8 * fs)
    plt.tick_params(which='minor', length=4, width=2, labelsize=0.8 * fs)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    # Set minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(20))

    # Adjust layout
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()

    # Save and show plot
    plt.savefig(f'ranking_{metric}.png', dpi=500, bbox_inches='tight')
    plt.show()
