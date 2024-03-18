import matplotlib.pyplot as plt

# Data from the image provided by the user
models = ['Best-SVR', 'Best-XGBoost', 'Stacking-SVR&XGB']
rmse = [0.4204, 0.4113, 0.4068]
mae = [0.3229, 0.3177, 0.3142]
cc = [0.8092, 0.8182, 0.8218]

# Adjusted function to plot bar chart using FixedLocator

import matplotlib.pyplot as plt


# Extracted hex color codes from the image provided by the user
color_codes = [

    '#EF7A6D',  # light coral
    '#F1D77E',  # light yellow
    '#5F97D2',  # steel blue

]

# Ensuring we have a color for each model, repeat the color list if needed
colors = color_codes * (len(models) // len(color_codes) + 1)


# Plot function with individual color for each bar
def plot_bar_chart_with_colors(models, data, metric_name, colors):
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.4  # Narrower bar width
    x = range(len(models))

    ax.bar(x, data, width=bar_width, color=colors[:len(models)])

    for i, model in enumerate(models):
        ax.bar(model, data[i], color=colors[i])

    # ax.set_xlabel('Model', fontsize=16)
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_title(f'{metric_name} of Models', fontsize=20)

    # Using FixedLocator to avoid the warning
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Dynamically setting the y-axis limits to enhance the visibility of small differences
    data_range = max(data) - min(data)
    upper_limit = max(data) + (data_range * 0.1)  # Adding 10% of range to upper limit for padding
    lower_limit = min(data) - (data_range * 0.1)  # Subtracting 10% of range to lower limit for padding
    ax.set_ylim(lower_limit, upper_limit)

    plt.tight_layout()
    plt.show()


# Plot RMSE, MAE, and CC with new colors
plot_bar_chart_with_colors(models, rmse, 'RMSE', colors[:len(models)])
plot_bar_chart_with_colors(models, mae, 'MAE', colors[:len(models)])
plot_bar_chart_with_colors(models, cc, 'CC', colors[:len(models)])
