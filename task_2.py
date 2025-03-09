import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Загрузка данных из файла 'iris.csv'
df = pd.read_csv('iris.csv')

# Расчёт площадей
df['Sepal.Area'] = df['Sepal.Length'] * df['Sepal.Width']
df['Petal.Area'] = df['Petal.Length'] * df['Petal.Width']
df['Total.Area'] = df['Sepal.Area'] + df['Petal.Area']

# ------------------ TASK 2 ------------------
from scipy import stats
from scipy.stats import norm, lognorm, shapiro

# Выбираем столбец для анализа
x = df['Total.Area'].values
n = len(x)
###1. Проверка гипотезы нормального распределения
## 1.1 Оценка параметров нормального распределения методом моментов
# Вычисление выборочных моментов
M1 = np.mean(x)  # Первый момент (выборочное среднее)
M2 = np.mean(x ** 2)  # Второй момент (среднее от квадратов)

# Метод моментов: решение системы
mu_hat = M1  # Оценка математического ожидания
sigma2_hat = M2 - M1 ** 2  # Оценка дисперсии (по формуле из картинки)
sigma_hat = np.sqrt(sigma2_hat)  # Оценка стандартного отклонения

# Определяем вектор параметров θ
theta = (mu_hat, sigma2_hat)

# Вывод результатов
print("\n=== Метод моментов: оценка параметров нормального распределения ===")
print(f"Первый момент (M1) = E[X] ≈ {M1:.4f}")
print(f"Второй момент (M2) = E[X²] ≈ {M2:.4f}")

print("\nОценки параметров:")
print(f"Оценка математического ожидания (μ̂) = {mu_hat:.4f}")
print(f"Оценка дисперсии (σ̂²) = {sigma2_hat:.4f}")
print(f"Оценка стандартного отклонения (σ̂) = {sigma_hat:.4f}")

print("\nВектор параметров θ:")
print(f"θ = ({mu_hat:.4f}, {sigma2_hat:.4f})")
## 1.2 Оценка гипотезы о нормальном распределении с помощью коэффициентов асимметрии и эксцесса
skewness = stats.skew(x)
kurtosis = stats.kurtosis(x, bias=True)

print(f"Коэффициент асимметрии (S): {skewness:.4f}")
print(f"Коэффициент эксцесса (K): {kurtosis:.4f}")

# Построение гистограммы с нормализацией плотности
plt.figure(figsize=(8, 5))
sns.histplot(x, bins=30, kde=True, stat="density", color='blue', alpha=0.6)

# Наложение нормальной кривой плотности с правильным масштабом
xs = np.linspace(min(x), max(x), 200)  # Генерируем точки по оси X
pdf_norm = norm.pdf(xs, loc=mu_hat, scale=sigma_hat)  # Правильный масштаб PDF

plt.plot(xs, pdf_norm, 'r-', linewidth=2, label=f'Нормальная PDF (μ={mu_hat:.2f}, σ={sigma_hat:.2f})')
plt.plot([], [], color='blue', linestyle='-', linewidth=2, label="KDE")

# Оформление графика
plt.title("Гистограмма всей выборки с наложенной нормальной кривой и KDE")
plt.xlabel("Суммарная площадь")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid()
plt.show()

# Вычислим логарифм правдоподобия для нормального распределения
#    log L = \sum_{i=1}^n log f(x_i | mu_hat, sigma_hat)
#    где f(x|mu,sigma) = 1/(sqrt(2π)*sigma) * exp(-(x-mu)^2/(2sigma^2))
#    Удобнее сразу вызвать norm.logpdf(...)

logpdf_values = norm.logpdf(x, loc=mu_hat, scale=sigma_hat)
log_likelihood = np.sum(logpdf_values)

print("\n=== Правдоподобие===")
print(f"log-likelihood = {log_likelihood:.3f}")

## 1.4 Проверка гипотезы о нормальности распределения с помощью критерия Шапиро
# Критерии проверки нормальности
#    - Shapiro-Wilk
shapiro_stat, shapiro_pval = shapiro(x)

print("\n=== Проверка гипотезы о нормальности ===")
print(f"Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_pval:.4f}")

if shapiro_pval < 0.05:
    print("По критерию Шапиро: можно отвергнуть гипотезу о нормальности (на уровне значимости 5%)")
else:
    print("По критерию Шапиро: нет оснований отвергать нормальность (p-value > 0.05)")


### 2. Проверка гипотезы логнормального распределения
# Логарифмируем данные
log_x = np.log(x)

# Оценка параметров логнормального распределения
mu_mom = np.mean(log_x)  # Выборочное среднее логарифмов
sigma2_mom = np.var(log_x, ddof=1)  # Выборочная дисперсия логарифмов (ddof=1 делает её несмещённой)

# Вывод результатов
print("=== Оценка параметров методом моментов ===")
print(f"μ (лог-среднее) = {mu_mom:.4f}")
print(f"σ² (лог-дисперсия) = {sigma2_mom:.4f}")

## 2.1 Оценка параметров логнормального распределения методом максимального правдоподобия
from scipy.optimize import minimize

#Функция для вычисления логарифма функции правдоподобия (Log-Likelihood)
def log_likelihood(params, x):
    mu, sigma2 = params
    # Логарифм правдоподобия для логнормального распределения
    log_x = np.log(x)
    n = len(x)
    log_likelihood_value = -(n - 1) * np.log(np.sqrt(sigma2)) - 0.5 * np.sum((log_x - mu)**2) / sigma2 - np.sum(np.log(x))
    return -log_likelihood_value  # Минмизируем, поэтому берем отрицательное значение

# Начальные приближения для mu и sigma^2
initial_params = [np.mean(np.log(x)), np.var(np.log(x), ddof=1)]  # Начальные приближения для mu и sigma^2

# Оценка параметров методом максимального правдоподобия (MLE) с использованием оптимизации
result = minimize(log_likelihood, initial_params, args=(x), bounds=[(None, None), (1e-6, None)])

mu_mle, sigma2_mle = result.x

# Вывод найденных параметров
theta_hat = (mu_mle, sigma2_mle)  # Вектор параметров θ

print("\n=== Оценка параметров методом максимального правдоподобия (MLE) ===")
print(f"Вектор параметров θ_MLE = ({mu_mle:.4f}, {sigma2_mle:.4f})")
print(f"Оценка параметра μ: {mu_mle:.4f}")
print(f"Оценка параметра σ² (дисперсия): {sigma2_mle:.4f}")

## 2.2 Статистические свойства найденной оценки
## Проверка смещённости
bias_mu = np.mean(np.log(x)) - mu_mle  # Логарифмируем данные и сравниваем со средним
bias_sigma2 = np.var(np.log(x), ddof=1) - sigma2_mle  # Вычисляем выборочную дисперсию логарифмов и сравниваем с оценкой σ²

print("\n=== Проверка смещённости оценок ===")
print(f"Смещение оценки μ: {bias_mu:.8f}")
print(f"Смещение оценки σ²: {bias_sigma2:.8f}")

if abs(bias_mu) < 1e-6:
    print("✅ Оценка μ является несмещённой.")
else:
    print("❌ Оценка μ является смещённой.")

if abs(bias_sigma2) < 1e-6:
    print("✅ Оценка σ² является несмещённой.")
else:
    print("❌ Оценка σ² является смещённой.")


##Состоятельность
# Оценка параметров методом максимального правдоподобия (MLE) с использованием оптимизации
def mle_estimate(x):
    initial_params = [np.mean(np.log(x)), np.var(np.log(x), ddof=1)]  # Начальные приближения для mu и sigma^2
    result = minimize(log_likelihood, initial_params, args=(x), bounds=[(None, None), (1e-6, None)])
    return result.x  # Возвращаем параметры (mu, sigma^2)

# Оценка параметров на полной выборке
mu_mle, sigma2_mle = mle_estimate(x)
theta_hat_full_sample = (mu_mle, sigma2_mle)

# Оценка состоятельности на подвыборках
sample_sizes = [10, 30, 50, 100, len(x)]  # Различные размеры подвыборок
mu_estimates = []
sigma2_estimates = []

for size in sample_sizes:
    sample = np.random.choice(x, size=size, replace=False)
    mu_hat_sample, sigma2_hat_sample = mle_estimate(sample)
    mu_estimates.append(mu_hat_sample)
    sigma2_estimates.append(sigma2_hat_sample)

# Построение графиков сходимости оценок
plt.figure(figsize=(12, 6))

# График для μ
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, mu_estimates, marker='o', linestyle='dashed', label="Оценки для μ")
plt.axhline(mu_mle, color='r', linestyle='--', label="MLE оценка μ")
plt.title("Сходимость оценки для μ")
plt.xlabel("Размер выборки")
plt.ylabel("Оценка μ")
plt.legend()
plt.grid()

# График для σ²
plt.subplot(1, 2, 2)
plt.plot(sample_sizes, sigma2_estimates, marker='o', linestyle='dashed', label="Оценки для σ²")
plt.axhline(sigma2_mle, color='r', linestyle='--', label="MLE оценка σ²")
plt.title("Сходимость оценки для σ²")
plt.xlabel("Размер выборки")
plt.ylabel("Оценка σ²")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Автоматическая проверка состоятельности
tolerance = 0.05  # 5% отклонение

mu_consistent = abs(mu_estimates[-1] - mu_mle) / abs(mu_mle) < tolerance
sigma2_consistent = abs(sigma2_estimates[-1] - sigma2_mle) / abs(sigma2_mle) < tolerance

print("\n=== Оценка состоятельности ===")
print(f"Оценка для самого большого n (μ̂): {mu_estimates[-1]:.4f}")
print(f"Оценка для самого большого n (σ̂²): {sigma2_estimates[-1]:.4f}")

if mu_consistent:
    print("✅ Оценка μ является состоятельной.")
else:
    print("❌ Оценка μ НЕ является состоятельной.")

if sigma2_consistent:
    print("✅ Оценка σ² является состоятельной.")
else:
    print("❌ Оценка σ² НЕ является состоятельной.")

## 2.3 Оценка гипотезы о логнормальном распределении с помощью коэффициентов асимметрии и эксцесса
# Вычисление коэффициентов
skewness = stats.skew(x)
kurtosis = stats.kurtosis(x, bias=True)

# Оценка параметров логнормального распределения (MLE)
shape_mle, loc_mle, scale_mle = lognorm.fit(x, floc=0)
sigma_hat = shape_mle
mu_hat = np.log(scale_mle)

# Вывод значений
print("\n=== Коэффициенты распределения ===")
print(f"Коэффициент асимметрии (S): {skewness:.4f}")
print(f"Коэффициент эксцесса (K): {kurtosis:.4f}")

# Построение графиков

## Гистограмма с KDE
plt.figure(figsize=(8, 5))
sns.histplot(x, bins=30, kde=True, stat="density", color='blue', alpha=0.6)

# Наложение теоретической кривой логнормального распределения
xs = np.linspace(min(x), max(x), 200)
pdf_lognorm = lognorm.pdf(xs, shape_mle, loc=loc_mle, scale=scale_mle)
plt.plot(xs, pdf_lognorm, 'r-', linewidth=2, label=f'Lognormal PDF (μ={mu_hat:.2f}, σ={sigma_hat:.2f})')
plt.plot([], [], color='blue', linestyle='-', linewidth=2, label="KDE")

plt.title("Гистограмма с KDE и логнормальным распределением")
plt.xlabel("Суммарная площадь")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid()
plt.show()

## 2.4 Вычисление логарифма правдоподобия для сравнения гипотез о законе распределения
# Логарифм правдоподобия для lognormal
logpdf_values = lognorm.logpdf(x, shape_mle, loc=loc_mle, scale=scale_mle)
log_likelihood = np.sum(logpdf_values)

print("\n=== Логарифм правдоподобия===")
print(f"log-likelihood = {log_likelihood:.3f}")
##2.5 Проверка гипотезы о логнормальности распределения с помощью кртирерия Шапиро
# Проверка гипотезы о логнормальном распределении
shapiro_stat, shapiro_pval = shapiro(log_x)

print("\n=== Проверка гипотезы о логнормальном распределении ===")
print(f"Shapiro-Wilk (проверка нормальности логарифма): statistic={shapiro_stat:.4f}, p-value={shapiro_pval:.4f}")

if shapiro_pval < 0.05:
    print("По критерию Шапиро: можно отвергнуть гипотезу о логнормальности (логарифм не нормален)")
else:
    print("По критерию Шапиро: нет оснований отвергать логнормальность")

# Визуализация: гистограмма + плотность
plt.figure(figsize=(8, 5))
sns.histplot(x, kde=False, stat='density', bins=15, color='skyblue', label='Data histogram')

# Теоретическая lognormal-плотность
xs = np.linspace(min(x), max(x), 200)
pdf_lognorm = lognorm.pdf(xs, shape_mle, loc=loc_mle, scale=scale_mle)
plt.plot(xs, pdf_lognorm, 'r-', label='Lognormal PDF (MLE)')

plt.title('Проверка гипотезы логнормального распределения')
plt.xlabel('Суммарная площадь (Total.Area)')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# Визуализация
plt.figure(figsize=(8, 5))
sns.histplot(log_x, kde=True, bins=15, color='skyblue', label='log(Total.Area)')
plt.title('Гистограмма логарифмов') #(если нормальная, то данные логнормальны)
plt.legend()
plt.show()


### 3. Теоретическое смещение, дисперсия, MSE, информация Фишера
# Теоретическое смещение оценок
bias_mu = 0  # Среднее несмещённое
bias_sigma_mle = 0  # MLE-дисперсия дает несмещенную оценку

# Теоретическая дисперсия оценок
var_mu = sigma2_mle**2 / n  # Дисперсия оценки μ
var_sigma2_mle = 2 * sigma2_mle**4 / n  # Дисперсия оценки σ^2 (MLE)

# MSE (среднеквадратическая ошибка)
mse_mu = var_mu  # Так как смещения нет
mse_sigma_mle = var_sigma2_mle + bias_sigma_mle**2

# Информация Фишера (по формулам)
I_mu_mu = n / sigma2_mle**2
I_sigma_sigma = n / (2 * sigma2_mle**4)

# 8. Вывод результатов
print("\n=== Оценки параметров ===")
print(f"Оценка μ (лог-среднее) = {mu_mle:.4f}")
print(f"Оценка σ (лог-стандартное отклонение) = {sigma2_mle:.4f}")

print("\n=== Теоретическое смещение ===")
print(f"Смещение μ (MLE) = {bias_mu:.4f}")
print(f"Смещение σ^2 (MLE) = {bias_sigma_mle:.4f}")

print("\n=== Теоретическая дисперсия ===")
print(f"Var(μ_hat) = {var_mu:.8f}")
print(f"Var(σ^2_hat) (MLE) = {var_sigma2_mle:.8f}")

print("\n=== MSE (среднеквадратическая ошибка) ===")
print(f"MSE(μ_hat) = {mse_mu:.8f}")
print(f"MSE(σ^2_hat) (MLE) = {mse_sigma_mle:.8f}")

print("\n=== Информация Фишера ===")
print(f"Fisher Information (μ-μ) = {I_mu_mu:.4f}")
print(f"Fisher Information (σ^2 - σ^2) = {I_sigma_sigma:.4f}")