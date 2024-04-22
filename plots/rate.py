import matplotlib.pyplot as plt

# Data
audio_interleaving_rate = [0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
mAP_05 = [29.0, 57.9, 59.6, 60.0, 59.6, 58.1, 39.4, 43.2, 42.3, 42.8, 44.7, 47.1]
mAP_06 = [23.9, 51.7, 53.9, 54.4, 54.5, 51.9, 35.6, 39.0, 38.7, 39.1, 40.6, 42.3]
mAP_07 = [18.8, 46.5, 48.7, 49.6, 49.4, 47.3, 32.7, 35.5, 35.4, 35.4, 36.7, 37.4]
mAP_08 = [13.6, 40.7, 43.1, 43.5, 43.1, 41.4, 28.8, 31.9, 31.2, 30.7, 31.3, 32.3]
mAP_09 = [8.8, 34.7, 36.5, 37.1, 37.1, 36.0, 25.4, 27.2, 26.6, 26.1, 27.3, 27.4]
avg_mAP = [35.8, 58.5, 59.9, 60.3, 60.0, 58.3, 39.6, 44.3, 44.2, 44.2, 46.0, 48.4]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(audio_interleaving_rate, mAP_05, label='mAP@0.5', marker='o', color='blue')
plt.plot(audio_interleaving_rate, mAP_06, label='mAP@0.6', marker='s', color='green')
plt.plot(audio_interleaving_rate, mAP_07, label='mAP@0.7', marker='^', color='red')
plt.plot(audio_interleaving_rate, mAP_08, label='mAP@0.8', marker='x', color='purple')
plt.plot(audio_interleaving_rate, mAP_09, label='mAP@0.9', marker='*', color='orange')
plt.plot(audio_interleaving_rate, avg_mAP, label='Avg. mAP', marker='d', linestyle='--', color='black')

# Formatting
plt.xlabel('Audio-Interleaving Rate (%)', fontsize=20)
plt.ylabel('mAP Score', fontsize=20)
plt.title('mAP Scores at Different Audio-Interleaving Rates', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20, loc='center', bbox_to_anchor=(0.5, 0.15), ncol=2)
plt.grid(True)
# plt.show()
plt.savefig('audio_interleaving_rate.png', bbox_inches='tight')
