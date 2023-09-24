import colorsys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio
import os

output_dir = "kmeans_images"
os.makedirs(output_dir, exist_ok=True)

point_size = 5

# Генерация случайных точек
def generate_random_points(num_points):
    return np.random.rand(num_points, 2) * 10  # Генерация точек в диапазоне [0, 10]

# Расчет координат центроидов в виде правильного многоугольника (например, треугольника)
def calculate_initial_centroids(k):
    angles = np.linspace(0, 2*np.pi, k, endpoint=False)
    centroids_x = 5 + 2 * np.cos(angles)
    centroids_y = 5 + 2 * np.sin(angles)
    return np.column_stack((centroids_x, centroids_y))

# Расстояние между двумя точками
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# K-means алгоритм
def kmeans(X, k, max_iters=100, save_steps=False):
    # Инициализация центроидов как правильного многоугольника вписанного в окружность
    centroids = calculate_initial_centroids(k)

    for iteration in range(max_iters):
        # Назначение кластеров
        labels = np.argmin(np.array([[distance(x, centroid) for centroid in centroids] for x in X]), axis=1)

        if save_steps:
            # Создание изображения для текущего шага
            plt.figure(figsize=(6, 6))
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', s=100)
            plt.title(f'Step {iteration + 1}')
            
            plt.savefig(os.path.join(output_dir, f'step_{iteration + 1}.png'))
            plt.close()

        # Обновление центроидов только для непустых кластеров
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroid for i, centroid in enumerate(centroids)])

         # Проверка на сходимость
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, labels

# Метод локтя для определения оптимального числа кластеров
def find_optimal_k(X, max_k=10):
    sse = []  # Сумма квадратов расстояний

    for k in range(1, max_k + 1):
        centroids, labels = kmeans(X, k)
        sse.append(np.sum((X - centroids[labels]) ** 2))

    # Визуализация метода локтя
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма квадратов расстояний')
    plt.title('Метод локтя')
    plt.show()

    # Возвращаем точки SSE для всех значений k
    return sse

if __name__ == "__main__":
    num_points = 300
    max_k = 10

    # Генерация случайных точек
    X = generate_random_points(num_points)

    # Определение оптимального числа кластеров и вывод графика метода локтя
    sse_values = find_optimal_k(X, max_k)

    # Определение числа кластеров для визуализации
    optimal_k = int(input("Введите оптимальное число кластеров для визуализации: "))

    # Выполнение K-means с оптимальным числом кластеров
    centroids, labels = kmeans(X, optimal_k, save_steps=True)

    # Вывод результатов
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red')
    plt.show()
