# ------------------ TASK 4 ------------------
import numpy as np


def theta_post_mean(b, mu, sigma, n, x):
    return (sigma ** 2 * np.sum(x) + b ** 2 * mu) / (n * b ** 2 + sigma ** 2)


m = 10000
n = 100
mu = 0
sigma = 1
b = 1

n_arr = np.random.randint(n - 5, n + 5, m)

true_theta = np.random.normal(mu, sigma)
bayesian_estimates = []
for i in n_arr:
    x = np.random.normal(true_theta, b, n)
    thetta = theta_post_mean(b, mu, sigma, n, x)
    bayesian_estimates.append(thetta)
bayesian_estimates = np.array(bayesian_estimates)
bayesian_estimates

import matplotlib.pyplot as plt


def plot_bayesian_estimates(theta_array, sample_sizes, true_theta):
    """
    Функция анализирует оценки параметров в зависимости от размера выборки.

    Параметры:
        theta_array (numpy.ndarray): массив оценок параметров (размерность N),
                                     где каждый элемент соответствует байесовской оценке θ.
        sample_sizes (numpy.ndarray): массив размеров выборки длины N,
                                      индексы которого совпадают с theta_array.
        true_theta (float): истинное значение параметра θ, для нанесения на графики.
    """
    unique_ns = np.unique(sample_sizes)

    for n in unique_ns:
        indices = np.where(sample_sizes == n)
        theta_n = theta_array[indices]

        # Печать статистики для каждой выборки
        print(f"\nРазмер выборки n = {n}:")
        print(f"θ: mean = {np.mean(theta_n):.3f}, std = {np.std(theta_n):.3f}, median = {np.median(theta_n):.3f}")

        # Построение графиков для θ
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Гистограмма для θ
        axs[0].hist(theta_n, bins=20, alpha=0.7, color='b', edgecolor='black')
        axs[0].axvline(true_theta, color='r', linestyle='--', label=f"Истинное θ = {true_theta}")
        axs[0].set_title(f"Histogram of θ (n={n})")
        axs[0].legend()

        # Box-plot для θ
        axs[1].boxplot(theta_n, vert=True)
        axs[1].axhline(true_theta, color='r', linestyle='--', label=f"Истинное θ = {true_theta}")
        axs[1].set_title(f"Box-plot of θ (n={n})")
        axs[1].legend()

        # Violin-plot для θ
        axs[2].violinplot(theta_n, showmeans=True)
        axs[2].axvline(true_theta, color='r', linestyle='--', label=f"Истинное θ = {true_theta}")
        axs[2].set_title(f"Violin plot of θ (n={n})")
        axs[2].legend()

        plt.tight_layout()
        plt.show()


plot_bayesian_estimates(bayesian_estimates, n_arr, true_theta)
