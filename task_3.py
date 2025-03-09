import pandas as pd

df = pd.read_csv('iris.csv')

df['Sepal.Area'] = df['Sepal.Length'] * df['Sepal.Width']
df['Petal.Area'] = df['Petal.Length'] * df['Petal.Width']
df['Total.Area'] = df['Sepal.Area'] + df['Petal.Area']

x = df['Total.Area'].values
n = len(x)

# ------------------ TASK 3 ------------------
from task_2 import theta_hat

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

m = 10000

n_arr = np.random.randint(n - 5, n + 5, m)

# n_arr
theta_hat
# (3.1200087026030667, 0.08466707254919735)

def get_thetta(x):
    initial_params = [np.mean(np.log(x)), np.var(np.log(x), ddof=1)]
    result = minimize(log_likelihood, initial_params, args=(x), bounds=[(None, None), (1e-6, None)])
    mu_mle, sigma2_mle = result.x
    return (mu_mle, sigma2_mle)


def log_likelihood(params, x):
    mu, sigma2 = params
    log_x = np.log(x)
    n = len(x)
    log_likelihood_value = -(n - 1) * np.log(np.sqrt(sigma2)) - 0.5 * np.sum((log_x - mu) ** 2) / sigma2 - np.sum(
        np.log(x))
    return -log_likelihood_value


thetta_list = []
mu = theta_hat[0]
sigma = theta_hat[1]
for i in n_arr:
    sample = np.random.lognormal(mu, sigma, size=i)
    thetta = get_thetta(sample)
    thetta_list.append(thetta)
thetta_list = np.array(thetta_list)
thetta_list


def analyze_theta(theta_array, sample_sizes):
    """
    Функция анализирует оценки параметров в зависимости от размера выборки.

    Параметры:
        theta_array (numpy.ndarray): массив оценок параметров shape (N, 2),
                                     где столбец 0 соответствует μ,
                                     а столбец 1 соответствует σ.
        sample_sizes (numpy.ndarray): массив размеров выборки длины N,
                                      индексы которого совпадают с theta_array.
    """
    unique_ns = np.unique(sample_sizes)
    for n in unique_ns:
        indices = np.where(sample_sizes == n)
        theta_n = theta_array[indices]

        # Извлекаем оценки для каждого параметра
        mu_vals = theta_n[:, 0]
        sigma_vals = theta_n[:, 1]

        # Выводим описательные статистики
        print(f"\nРазмер выборки n = {n}:")
        print(f"μ: mean = {np.mean(mu_vals):.3f}, std = {np.std(mu_vals):.3f}, median = {np.median(mu_vals):.3f}")
        print(
            f"σ: mean = {np.mean(sigma_vals):.3f}, std = {np.std(sigma_vals):.3f}, median = {np.median(sigma_vals):.3f}")

        # Построение графиков для μ
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Гистограмма для μ
        axs[0].hist(mu_vals, bins=20)
        axs[0].set_title(f"Histogram of μ (n={n})")

        # Box-plot для μ
        axs[1].boxplot(mu_vals, vert=True)
        axs[1].set_title(f"Box-plot of μ (n={n})")

        # Violin-plot для μ
        axs[2].violinplot(mu_vals, showmeans=True)
        axs[2].set_title(f"Violin plot of μ (n={n})")

        plt.tight_layout()
        plt.show()

        # Построение графиков для σ
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Гистограмма для σ
        axs[0].hist(sigma_vals, bins=20)
        axs[0].set_title(f"Histogram of σ (n={n})")

        # Box-plot для σ
        axs[1].boxplot(sigma_vals, vert=True)
        axs[1].set_title(f"Box-plot of σ (n={n})")

        # Violin-plot для σ
        axs[2].violinplot(sigma_vals, showmeans=True)
        axs[2].set_title(f"Violin plot of σ (n={n})")

        plt.tight_layout()
        plt.show()


analyze_theta(thetta_list, n_arr)
