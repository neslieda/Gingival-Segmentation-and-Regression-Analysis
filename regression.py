import pandas as pd

# Load the data from the provided Excel file
file_path = 'output.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the data to understand its structure
data.head()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to perform regression for each tooth and return performance metrics
def perform_regression(x, y):
    # Reshape x for sklearn
    x = x.values.reshape(-1, 1)
    # Create and fit the model
    model = LinearRegression()
    model.fit(x, y)
    # Predictions
    y_pred = model.predict(x)
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    # Return model, predictions, and metrics
    return model, y_pred, mse, r2

# Create dictionaries to store models, predictions, and metrics
models = {}
predictions = {}
metrics = {}

# List of teeth
teeth = ['1', '2', '3', '4', '5', '6']

# Perform regression for each tooth
for tooth in teeth:
    x = data[f'{tooth}p']
    y = data[f'{tooth}_r']
    model, y_pred, mse, r2 = perform_regression(x, y)
    models[tooth] = model
    predictions[tooth] = y_pred
    metrics[tooth] = {'MSE': mse, 'R2': r2}

# Plot the results
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

for i, tooth in enumerate(teeth):
    axs[i].scatter(data[f'{tooth}p'], data[f'{tooth}_r'], color='blue', label='Actual')
    axs[i].plot(data[f'{tooth}p'], predictions[tooth], color='red', linewidth=2, label='Predicted')
    axs[i].set_title(f'Diş {tooth} - MSE: {metrics[tooth]["MSE"]:.2f}, R2: {metrics[tooth]["R2"]:.2f}')
    axs[i].set_xlabel('Piksel Değeri')
    axs[i].set_ylabel('Gerçek Ölçüm Değeri')
    axs[i].legend()

plt.tight_layout()
plt.show()

print(metrics)

import numpy as np

# Korelasyon katsayılarını hesaplama
correlations = {}
for tooth in teeth:
    x = data[f'{tooth}p']
    y = data[f'{tooth}_r']
    correlation = np.corrcoef(x, y)[0, 1]
    correlations[tooth] = correlation

# Korelasyon katsayılarını ve diğer metrikleri birleştirerek sonuçları oluşturma
results = pd.DataFrame({
    'MSE': [metrics[tooth]['MSE'] for tooth in teeth],
    'R2': [metrics[tooth]['R2'] for tooth in teeth],
    'Correlation': [correlations[tooth] for tooth in teeth]
}, index=[f'Diş {tooth}' for tooth in teeth])

# Sonuçları görüntüleme
print(results)

# Calculate Pearson correlation coefficients for each tooth
for tooth in teeth:
    x = data[f'{tooth}p']
    y = data[f'{tooth}_r']
    correlation = np.corrcoef(x, y)[0, 1]
    results.loc[f'Diş {tooth}', 'Correlation'] = correlation

print(results)

# Her diş için scatter plot oluşturma ve görüntüleme
for tooth in teeth:
    plt.figure()
    plt.scatter(data[f'{tooth}p'], data[f'{tooth}_r'], color='blue', label='Klinik Ölçüm')
    plt.plot(data[f'{tooth}p'], predictions[tooth], color='red', linewidth=2, label='AI Tahmini')
    plt.title(f'Diş {tooth} - MSE: {metrics[tooth]["MSE"]:.2f}, R2: {metrics[tooth]["R2"]:.2f}, Korelasyon: {correlations[tooth]:.2f}')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Gerçek Ölçüm Değeri')
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def bland_altman_plot(data1, data2, title):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    plt.scatter(mean, diff, color='blue', alpha=0.5)
    plt.axhline(md, color='red', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Measurements')
    plt.title(title)
    plt.show()


# Her diş için Bland-Altman grafiği oluşturma
for tooth in teeth:
    bland_altman_plot(data[f'{tooth}_r'], predictions[tooth], f'Diş {tooth} Bland-Altman Grafiği')


for tooth in teeth:
    plt.figure()
    plt.scatter(data[f'{tooth}p'], data[f'{tooth}_r'], color='blue', label='Klinik Ölçüm')
    plt.plot(data[f'{tooth}p'], predictions[tooth], color='red', linewidth=2, label='AI Tahmini')
    plt.title(f'Diş {tooth} - MSE: {metrics[tooth]["MSE"]:.2f}, R2: {metrics[tooth]["R2"]:.2f}, Korelasyon: {correlations[tooth]:.2f}')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Gerçek Ölçüm Değeri')
    plt.legend()
    plt.show()

# Bland-Altman analizi için gerekli kütüphaneleri ekleyelim
import statsmodels.api as sm


# Bland-Altman grafiği oluşturma fonksiyonu
def bland_altman_plot(data1, data2, ax=None):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    if ax is None:
        ax = plt.gca()
    ax.scatter(mean, diff, color='blue', alpha=0.5)
    ax.axhline(md, color='red', linestyle='--')
    ax.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    ax.set_xlabel('Mean of Two Measurements')
    ax.set_ylabel('Difference Between Measurements')
    return ax


# Her diş için Bland-Altman grafiği oluşturma ve kaydetme
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

for i, tooth in enumerate(teeth):
    ax = bland_altman_plot(data[f'{tooth}_r'], predictions[tooth], ax=axs[i])
    ax.set_title(f'Diş {tooth}')

plt.tight_layout()
plt.show()

import scipy.stats as stats

# Paired t-test ve Wilcoxon signed-rank test sonuçlarını saklamak için bir DataFrame oluşturma
test_results = pd.DataFrame(index=[f'Diş {tooth}' for tooth in teeth],
                            columns=['Paired t-test p-value', 'Wilcoxon signed-rank p-value'])

# Her diş için testleri uygulama
for tooth in teeth:
    # Gerçek ölçümler ve tahminler
    actual_values = data[f'{tooth}_r']
    predicted_values = predictions[tooth]

    # Paired t-test
    t_test_stat, t_test_p_value = stats.ttest_rel(actual_values, predicted_values)

    # Wilcoxon signed-rank test
    wilcoxon_stat, wilcoxon_p_value = stats.wilcoxon(actual_values, predicted_values)

    # Sonuçları DataFrame'e ekleme
    test_results.loc[f'Diş {tooth}', 'Paired t-test p-value'] = t_test_p_value
    test_results.loc[f'Diş {tooth}', 'Wilcoxon signed-rank p-value'] = wilcoxon_p_value

# Test sonuçlarını görüntüleme
print(test_results)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes


# Passing-Bablok regresyonu fonksiyonu
def passing_bablok(x, y):
    # Theil-Sen regresyonu
    res = theilslopes(y, x, 0.95)

    slope = res[0]
    intercept = res[1]
    lower_slope = res[2]
    upper_slope = res[3]

    # Regresyon doğrusunu çizme
    plt.figure()
    plt.scatter(x, y, color='blue', label='Klinik Ölçümler vs. AI Tahminleri')
    plt.plot(x, slope * x + intercept, color='red', label=f'Regresyon Doğrusu\ny = {slope:.2f}x + {intercept:.2f}')
    plt.fill_between(x, lower_slope * x + intercept, upper_slope * x + intercept, color='red', alpha=0.2)
    plt.title('Passing-Bablok Regresyonu')
    plt.xlabel('Klinik Ölçümler')
    plt.ylabel('AI Tahminleri')
    plt.legend()
    plt.show()

    return {
        'slope': slope,
        'intercept': intercept,
        'lower_slope': lower_slope,
        'upper_slope': upper_slope
    }


# Her diş için Passing-Bablok regresyonunu uygulama
results = {}
for tooth in teeth:
    x = data[f'{tooth}p']
    y = data[f'{tooth}_r']
    result = passing_bablok(x, y)
    results[tooth] = result
    print(f'Diş {tooth} için Passing-Bablok sonuçları:')
    print(result)
