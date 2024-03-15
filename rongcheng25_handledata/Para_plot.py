import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['GS-SVR', 'Bayes-SVR', 'GS-XGB', 'Bayes-XGB', 'GS-RF', 'Bayes-RF']
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4']

# RMSE, MAD, Pearson Correlation for each day
rmse_data = {
    'Day 1': [0.3945, 0.3456, 0.3656, 0.3597, 0.3651, 0.3839],
    'Day 2': [0.3222, 0.3593, 0.3253, 0.3400, 0.3567, 0.3198],
    'Day 3': [0.3709, 0.3165, 0.3150, 0.3039, 0.3286, 0.3572],
    'Day 4': [0.4048, 0.4153, 0.3940, 0.3959, 0.4005, 0.4565]
}

mad_data = {
    'Day 1': [0.3323, 0.2823, 0.3012, 0.2962, 0.2974, 0.3199],
    'Day 2': [0.2495, 0.3031, 0.2643, 0.2741, 0.2866, 0.2631],
    'Day 3': [0.3176, 0.2656, 0.2619, 0.2486, 0.2602, 0.2834],
    'Day 4': [0.2790, 0.2783, 0.2545, 0.2604, 0.2539, 0.3097]
}

pearson_data = {
    'Day 1': [0.9079, 0.9113, 0.9081, 0.9078, 0.9064, 0.9159],
    'Day 2': [0.7992, 0.8192, 0.8197, 0.8141, 0.8131, 0.8065],
    'Day 3': [0.9439, 0.9288, 0.9383, 0.9367, 0.9111, 0.9283],
    'Day 4': [0.9323, 0.8958, 0.9163, 0.9159, 0.8960, 0.9055]
}


# Plotting function modified to adjust y-axis limits for clarity
def plot_metrics(day, metrics, title):
    x = np.arange(len(models))
    width = 0.3

    # Compute the min and max for the current metric to adjust y-axis
    metric_min = min(metrics[day])
    metric_max = max(metrics[day])
    y_margin = (metric_max - metric_min) * 0.1  # Add a 10% margin

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, metrics[day], width, label=day)

    ax.set_ylabel('Scores')
    ax.set_title(f'{title} by model and {day}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([metric_min - y_margin, metric_max + y_margin])  # Adjusting y-axis limits
    ax.legend()

    fig.tight_layout()

    plt.show()

# Plotting for each day
for day in days:
    plot_metrics(day, rmse_data, 'RMSE')
    plot_metrics(day, mad_data, 'MAD')
    plot_metrics(day, pearson_data, 'Pearson Correlation')
