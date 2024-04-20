from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from svm_optimization import svm_optimization

# fetch dataset
dry_bean = fetch_ucirepo(id=602)

# data (as pandas dataframes)
X = dry_bean.data.features
y = dry_bean.data.targets
data = pd.concat([X, y], axis=1)
print(data.head())

# Creating samples
samples = [data.sample(1000) for _ in range(10)]

# Variables keep record of best_values
overall_best_accuracy = 0
overall_best_sample_past_accuracies = []
best_accuracies = []
sample_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
best_kernels = []
best_gammas = []
best_Cs = []
iterations = 100

# Running SVM_Optimization function for each Sample
for i in samples:
    [best_params, best_accuracy, accuracy_values] = svm_optimization(i , iterations)
    best_kernels.append(best_params['kernel'])
    best_gammas.append(round(best_params['gamma'], 2))
    best_Cs.append(round(best_params['C'], 2))
    best_accuracies.append(best_accuracy)

    if best_accuracy > overall_best_accuracy:
        overall_best_sample_past_accuracies = accuracy_values
        overall_best_accuracy = best_accuracy

print(overall_best_accuracy)
result_table_df = pd.DataFrame([sample_names, best_accuracies, best_kernels, best_Cs, best_gammas])

df_transposed = result_table_df.T
df_transposed.columns = ['Sample', 'Best Accuracy', 'Best Kernel', 'Best C Vals.', 'Best Gamma Vals.']

# Create separate figures for the table and plot
fig_table, ax_table = plt.subplots(figsize=(8, 5))
fig_plot, ax_plot = plt.subplots(figsize=(6, 4))

# Creating The Analysis Table
table = ax_table.table(cellText=df_transposed.values,
          colLabels=df_transposed.columns,
          loc='center')
ax_table.axis('off')  # Hide the axes
table.auto_set_font_size(False)
table.set_fontsize(9)
ax_table.set_title('Analysis Table')

# Creating The Analysis Plot for Sample with highest Best Accuracy
ax_plot.plot(range(1, len(overall_best_sample_past_accuracies) + 1), overall_best_sample_past_accuracies, marker='o')
ax_plot.set_title('Convergence Plot')
ax_plot.set_xlabel('Iteration')
ax_plot.set_ylabel('Accuracy')
plt.show()
