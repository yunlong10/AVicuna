import matplotlib.pyplot as plt
import numpy as np

# Data
model_labels = np.array(['Valley', 'VideoChat', 'Video-ChatGPT', 'VTimeLLM', 
                         'PandaGPT', 'Macaw-LLM', 'Video-LLaMA', 'AV-LLM', 'AVicuna (ours)'])
dataset_labels = np.array(['AVSD', 'MUSIC-AVQA', 'MSVD-QA', 'MSRVTT-QA', 
                           'ActivityNet-QA', 'ActivityNet Captions', 'UnAV-100'])

model_stats = np.array([
    [0, 0, 65.4, 45.7, 42.9, 0, 0],         # Valley
    [0, 0, 56.3, 45, 26.5, 0, 0],            # VideoChat
    [0, 0, 64.9, 49.3, 35.2, 18.9, 0],       # Video-ChatGPT
    [0, 0, 69.8, 58.8, 45.5, 26.6, 0],       # VTimeLLM
    [26.1, 33.7, 46.7, 23.7, 11.2, 0, 0],    # PandaGPT
    [34.3, 31.8, 42.1, 25.5, 14.5, 0, 0],    # Macaw-LLM
    [36.7, 36.6, 51.6, 29.6, 12.4, 6.5, 2.3], # Video-LLaMA
    [52.6, 45.2, 67.3, 53.7, 47.2, 0, 0],    # AV-LLM
    [53.1, 49.6, 70.2, 59.7, 53, 28.1, 60.3] # AVicuna (ours)
])

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Increase font size for the chart text but keep the overall figure smaller
plt.rcParams.update({'font.size': 18})

# Define a new color palette with distinct colors for each model
new_color_palette = plt.cm.tab10(np.linspace(0, 1, len(model_labels)))
new_color_palette[3] = (50/255, 197/255, 212/255, 1.0)

new_color_palette[8] = (252/255, 205/255, 200/255, 1.0)
# print(new_color_palette)

# Split the circle into even parts and save the angles
angles = np.linspace(0, 2 * np.pi, len(dataset_labels), endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Determine the maximum values for each axis
axis_ranges = np.max(model_stats, axis=0)

# Draw one axe per variable and add labels
ax.set_thetagrids(np.degrees(angles[:-1]), dataset_labels, fontsize=20)

# Plot data and fill with color for each model using the new color palette
for idx, model_data in enumerate(model_stats):
    # Normalize the data for each axis
    normalized_data = model_data / axis_ranges
    normalized_data = np.concatenate((normalized_data, [normalized_data[0]]))  # ensure the data loops properly
    ax.plot(angles, normalized_data, label=model_labels[idx], color=new_color_palette[idx], linewidth=1)
    ax.fill(angles, normalized_data, color=new_color_palette[idx], alpha=0.25)

for i in range(len(dataset_labels)):
    axis_values = [stats[i] for stats in model_stats]
    # Sort the axis values while keeping track of the indices to match colors later
    sorted_values = sorted((value, idx) for idx, value in enumerate(axis_values))
    last_value = -1
    for value, idx in sorted_values:
        normalized_value = value / axis_ranges[i]
        angle = angles[i]
        # Check the distance to the last value; if too close, skip labeling
        if last_value < 0 or normalized_value - last_value > 0.1:
            ax.text(angle, normalized_value, str(value), ha='center', va='center', fontsize=16, color=new_color_palette[idx])
            last_value = normalized_value

# Add a legend with a 3x3 grid layout below the chart to prevent overlap and for aesthetic presentation
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=18, ncol=3)

# Add a title with a larger font size
# ax.set_title("Model Performance on Different Datasets", size=24, color='blue', y=1.1)

# Remove the y-axis labels (rings) to avoid confusion, as each axis has a different scale
ax.set_yticklabels([])

# Show the plot
# plt.show()
plt.savefig('radar_chart.png', bbox_inches='tight')
