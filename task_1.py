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

# Подсчёт количества экземпляров каждого вида
species_counts = df['Species'].value_counts()
print("Количество экземпляров каждого вида:")
print(species_counts)

# Количество экземпляров каждого вида:
# Species
# setosa        50
# versicolor    50
# virginica     50
# Name: count, dtype: int64

# Расчёт статистик для всей совокупности
total_area = df['Total.Area']
mean_total = total_area.mean()
var_total = total_area.var()
median_total = total_area.median()
quantile_0_4_total = total_area.quantile(0.4)

print("\nСтатистики для всей выборки (total_area):")
print(f"Выборочное среднее: {mean_total:.3f}")
print(f"Выборочная дисперсия: {var_total:.3f}")
print(f"Выборочная медиана: {median_total:.3f}")
print(f"Выборочная квантиль (p=0.4): {quantile_0_4_total:.3f}")

# Статистики для всей выборки (total_area):
# Выборочное среднее: 23.617
# Выборочная дисперсия: 47.909
# Выборочная медиана: 22.500
# Выборочная квантиль (p=0.4): 20.316

# Расчёт статистик по видам
group_stats = df.groupby('Species')['Total.Area'].agg(
    mean='mean',
    variance=lambda x: x.var(),
    median='median',
    quantile_0_4=lambda x: x.quantile(0.4)
)
print("\nСтатистики по видам:")
print(group_stats)

# Статистики по видам:
#                mean   variance  median  quantile_0_4
# Species
# setosa      17.6234   8.940060  17.660        16.736
# versicolor  22.2466  15.834109  22.210        21.142
# virginica   30.9808  27.004918  31.475        29.716

# ВИЗУАЛИЗАЦИИ

# Эмпирическая функция распределения для всей выборки
x_sorted = np.sort(total_area)
y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)  # Получаем доли
plt.figure(figsize=(8, 6))
plt.step(x_sorted, y, where='post')
plt.xlabel('Суммарная площадь')
plt.ylabel('Эмпирическая функция распределения')
plt.title('Эмпирическая функция распределения для всей выборки')
plt.grid(True)
plt.show()

# Гистограмма для всей выборки
plt.figure(figsize=(8, 6))
plt.hist(total_area, bins=20, edgecolor='black', alpha=0.7)  # bins - кол.во интервалов (колонок), alpha - прозрачность
plt.xlabel('Суммарная площадь')
plt.ylabel('Частота')
plt.title('Гистограмма суммарной площади для всей выборки')
plt.grid(True)
plt.show()

# Построение box-plot для всей выборки
plt.figure(figsize=(8, 6))  # создать окно с размерами 8 на 6 в дюймах
sns.boxplot(y=df['Total.Area'])
plt.ylabel('Суммарная площадь')
plt.title('Box-plot суммарной площади для всей выборки')
plt.grid(True)
plt.show()

# Box-plot суммарной площади по видам
plt.figure(figsize=(8, 6))  # создать окно с размерами 8 на 6 в дюймах
sns.boxplot(x='Species', y='Total.Area', data=df)
plt.xlabel('Вид')
plt.ylabel('Суммарная площадь (total_area)')
plt.title('Box-plot суммарной площади по видам')
plt.show()

# Эмпирическая функция распределения для каждого вида
plt.figure(figsize=(8, 6))  # создать окно с размерами 8 на 6 в дюймах
species_list = df['Species'].unique()
for species in species_list:
    data = df[df['Species'] == species]['Total.Area']  # Выбираем значения только вида species
    x_sorted = np.sort(data)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    plt.step(x_sorted, y, where='post', label=species)
plt.xlabel('Суммарная площадь')
plt.ylabel('Эмпирическая функция распределения')
plt.title('Эмпирическая функция распределения для каждого вида')
plt.legend()
plt.grid(True)
plt.show()

# Гистограммы для каждого вида
fig, axes = plt.subplots(1, len(species_list), figsize=(15, 5), sharey=True)
for ax, species in zip(axes, species_list):
    data = df[df['Species'] == species]['Total.Area']  # Выбираем значения только вида species
    ax.hist(data, bins=15, edgecolor='black', alpha=0.7)
    ax.set_title(species)
    ax.set_xlabel('Суммарная площадь')
    ax.set_ylabel('Частота')
plt.suptitle('Гистограммы суммарной площади по видам')
plt.tight_layout()
plt.show()
