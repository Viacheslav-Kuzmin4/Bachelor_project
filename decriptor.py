import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from gyrator import Gyrator, parse_angle_input, parse_H_input
from matplotlib.colors import rgb_to_hsv

# ==== ПАРАМЕТРЫ ====
mag_path =  r"C:\Users\95384\Desktop\Practice\gyrator_output_magnitude.png"
phase_path = r"C:\Users\95384\Desktop\Practice\gyrator_output_phase.png"
output_path = r"C:\Users\95384\Desktop\Practice\restored.png"

def load_image_as_array(path):
    """Загрузка изображения как массива."""
    try:
        return np.array(Image.open(path)).astype(np.float32) / 255.0
    except FileNotFoundError:
        print(f"Ошибка: файл не найден: {path}")
        print(f"Создаю тестовое изображение...")
        # Создаем тестовое изображение 256x256
        size = 256
        test_image = np.zeros((size, size, 3))
        for i in range(3):
            test_image[:, :, i] = np.random.rand(size, size)
        return test_image

def visualize_color_image(img, title):
    """Визуализация цветного изображения."""
    plt.figure(figsize=(8, 8))
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()

def normalize_image(data):
    """Нормализация изображения к диапазону [0, 1]."""
    if data.size == 0:
        return data
    data_min = np.min(data)
    data_max = np.max(data)
    if abs(data_max - data_min) < 1e-8:
        return np.zeros_like(data) if data_min == 0 else np.ones_like(data)
    return (data - data_min) / (data_max - data_min)

def combine_magnitude_phase(mag_rgb, phase_rgb):
    """
    Восстановление комплексного изображения из RGB-модуля и RGB-фазы.
    """
    # Извлечение фазы из оттенка (hue)
    phase_hsv = rgb_to_hsv(phase_rgb)
    hue = phase_hsv[..., 0]  # Hue канал содержит нормализованную фазу [0, 1]
    
    # Преобразование hue в фазу [-π, π]
    phase = hue * 2 * np.pi - np.pi
    
    # Восстановление комплексного изображения для каждого канала
    complex_channels = []
    
    for i in range(3):
        mag = mag_rgb[:, :, i]
        
        # Для восстановления модуля используем обратное преобразование нормализации
        # Поскольку мы знаем, что модуль был нормализован к [0, 1],
        # мы можем использовать обратное линейное преобразование
        # Для простоты будем считать, что масштаб сохраняется
        
        # Комплексное число: модуль * exp(i*фаза)
        complex_channel = mag * np.exp(1j * phase)
        complex_channels.append(complex_channel)
    
    return np.stack(complex_channels, axis=2)

def get_transform_parameters():
    """Получение параметров преобразования от пользователя."""
    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ ОБРАТНОГО ПРЕОБРАЗОВАНИЯ")
    print("=" * 60)
    
    print("\nВыберите способ ввода параметра преобразования:")
    print("[1] Ввести угол α (в радианах или с pi)")
    print("[2] Ввести параметр H = tan(α)")
    
    input_type = input("Ваш выбор (1 или 2): ").strip()
    
    if input_type == '1':
        print("\nВведите угол α (использовавшийся при прямом преобразовании):")
        print("Примеры: '0.25' → 0.25π, 'pi/4' → π/4, '0.5*pi' → 0.5π, '45deg' → 45°")
        alpha_str = input("α = ").strip()
        
        try:
            alpha = parse_angle_input(alpha_str)
            if np.isinf(alpha):
                H = float('inf') if alpha > 0 else float('-inf')
            else:
                H = np.tan(alpha)
            print(f"α = {alpha:.6f} рад = {alpha/np.pi:.6f}π")
            print(f"H = tan(α) = {H}")
        except Exception as e:
            print(f"Ошибка: {e}")
            return None, None, None
            
    elif input_type == '2':
        print("\nВведите параметр H (использовавшийся при прямом преобразовании):")
        print("Примеры: '0' → тождественное, '1' → α=π/4, 'inf' → α=π/2 (Фурье)")
        H_str = input("H = ").strip()
        
        try:
            H = parse_H_input(H_str)
            if np.isinf(H):
                alpha = np.pi/2 * np.sign(H)
            elif H == 0:
                alpha = 0.0
            else:
                alpha = np.arctan(H)
            print(f"H = {H}")
            print(f"α = arctan(H) = {alpha:.6f} рад = {alpha/np.pi:.6f}π")
        except Exception as e:
            print(f"Ошибка: {e}")
            return None, None, None
    else:
        print("\nИспользуется значение по умолчанию: H = 0")
        alpha = 0.0
        H = 0.0
    
    return alpha, H

# ==== ОСНОВНАЯ ПРОГРАММА ====
def main():
    print("=" * 60)
    print("ОБРАТНОЕ ГИРАТОРНОЕ ПРЕОБРАЗОВАНИЕ ЦВЕТНОГО ИЗОБРАЖЕНИЯ")
    print("=" * 60)
    
    # ==== ЗАГРУЗКА ДАННЫХ ====
    print(f"\nЗагрузка модуля: {mag_path}")
    mag_rgb = load_image_as_array(mag_path)
    
    print(f"Загрузка фазы: {phase_path}")
    phase_rgb = load_image_as_array(phase_path)
    
    print(f"Размер модуля: {mag_rgb.shape}")
    print(f"Размер фазы: {phase_rgb.shape}")
    
    # Проверяем, что изображения имеют одинаковый размер
    if mag_rgb.shape[:2] != phase_rgb.shape[:2]:
        print("Ошибка: изображения модуля и фазы имеют разные размеры!")
        print(f"Модуль: {mag_rgb.shape}, Фаза: {phase_rgb.shape}")
        return
    
    # Определяем размер изображения из загруженных данных
    height, width, channels = mag_rgb.shape
    if height != width:
        print(f"Предупреждение: изображение не квадратное ({height}x{width})")
        size = min(height, width)
        print(f"Будет использован минимальный размер: {size}")
        # Обрезаем изображения до квадрата
        mag_rgb = mag_rgb[:size, :size, :]
        phase_rgb = phase_rgb[:size, :size, :]
    else:
        size = height
    
    print(f"\nРазмер изображения автоматически определен: {size}x{size}")
    
    # Получение параметров преобразования
    alpha, H = get_transform_parameters()
    if alpha is None:
        print("Не удалось получить параметры преобразования.")
        return
    
    # Для обратного преобразования используем угол -alpha
    alpha_inv = -alpha
    if np.isinf(H):
        H_inv = -H
    elif H == 0:
        H_inv = 0
    else:
        H_inv = -H
    
    print(f"\nПАРАМЕТРЫ ОБРАТНОГО ПРЕОБРАЗОВАНИЯ:")
    print(f"  Исходный угол α: {alpha:.6f} рад = {alpha/np.pi:.6f}π")
    print(f"  Обратный угол: {alpha_inv:.6f} рад = {alpha_inv/np.pi:.6f}π")
    print(f"  Исходный H: {H}")
    print(f"  Обратный H: {H_inv}")
    print(f"  Размер изображения: {size}x{size}")
    
    # ==== ВОССТАНОВЛЕНИЕ КОМПЛЕКСНОГО ИЗОБРАЖЕНИЯ ====
    print("\nВосстановление комплексного изображения из модуля и фазы...")
    complex_image = combine_magnitude_phase(mag_rgb, phase_rgb)
    
    # ==== СОЗДАНИЕ ГИРАТОРА ====
    print(f"\nСоздание гиратора с размером {size}x{size}...")
    gyrator = Gyrator(size=size, scale=1.0)
    
    # ==== ПРИМЕНЕНИЕ ОБРАТНОГО ГИРАТОРНОГО ПРЕОБРАЗОВАНИЯ ====
    print("\nПрименение обратного гираторного преобразования...")
    restored_channels = []
    
    for i, channel_name in enumerate(['R', 'G', 'B']):
        print(f"Обработка канала {channel_name}...")
        
        channel_complex = complex_image[:, :, i]
        
        # Применение обратного гираторного преобразования
        if np.isinf(H_inv):
            # Особые случаи для бесконечного H
            if H_inv > 0:
                # Преобразование Фурье
                restored = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(channel_complex)))
            else:
                # Обратное преобразование Фурье
                restored = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(channel_complex)))
        else:
            # Обычное обратное гираторное преобразование
            restored = gyrator.transform(channel_complex, alpha_inv)
        
        # Берем модуль результата (фазу отбрасываем для визуализации)
        restored_channels.append(np.abs(restored))
    
    # ==== СОЗДАНИЕ ВОССТАНОВЛЕННОГО RGB ИЗОБРАЖЕНИЯ ====
    restored_rgb = np.stack(restored_channels, axis=2)
    
    # Нормализация каждого канала отдельно
    for i in range(3):
        restored_rgb[:, :, i] = normalize_image(restored_rgb[:, :, i])
    
    # ==== ВИЗУАЛИЗАЦИЯ ====
    visualize_color_image(mag_rgb, "Входной модуль")
    visualize_color_image(phase_rgb, "Входная фаза")
    visualize_color_image(restored_rgb, f"Восстановленное изображение (α={alpha/np.pi:.3f}π)")
    
    # ==== СОХРАНЕНИЕ РЕЗУЛЬТАТА ====
    # Создаем директорию если она не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    clipped = np.clip(restored_rgb, 0, 1)
    img = Image.fromarray((clipped * 255).astype(np.uint8), mode='RGB')
    img.save(output_path)
    
    print(f"Изображение сохранено")
    
    # ==== ОЦЕНКА КАЧЕСТВА ВОССТАНОВЛЕНИЯ ====
    print("\nОЦЕНКА КАЧЕСТВА ВОССТАНОВЛЕНИЯ:")
    print(f"  Размер восстановленного изображения: {restored_rgb.shape}")
    print(f"  Диапазон значений: [{np.min(restored_rgb):.3f}, {np.max(restored_rgb):.3f}]")
    print(f"  Средняя яркость: {np.mean(restored_rgb):.3f}")

if __name__ == "__main__":
    main()