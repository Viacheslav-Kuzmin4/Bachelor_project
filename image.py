import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from gyrator import Gyrator, parse_angle_input, parse_H_input

# ==== ПАРАМЕТРЫ ====
# Используйте один из следующих вариантов:
image_path = r"c:\Users\95384\Desktop\Practice\Витраж.jpg"
output_magnitude_path = r"c:\Users\95384\Desktop\Practice\gyrator_output_magnitude.png"
output_phase_path = r"c:\Users\95384\Desktop\Practice\gyrator_output_phase.png"

resize_to = 1024

def load_color_image(path, size=None):
    """Загрузка цветного изображения и приведение к нужному размеру."""
    try:
        img = Image.open(path).convert('RGB')
        if size:
            img = img.resize((size, size), Image.BICUBIC)
        return np.array(img).astype(np.float32) / 255.0
    except FileNotFoundError:
        print(f"Ошибка: файл не найден: {path}")
        print(f"Текущая рабочая директория: {os.getcwd()}")
        # Создаем тестовое изображение если файл не найден
        print("Создаю тестовое изображение...")
        test_image = np.zeros((size, size, 3))
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        test_image[mask] = [1, 0.5, 0]  # Оранжевый круг
        return test_image

def save_color_image(data, path):
    """Сохранение RGB изображения."""
    # Создаем директорию если она не существует
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    clipped = np.clip(data, 0, 1)
    img = Image.fromarray((clipped * 255).astype(np.uint8), mode='RGB')
    img.save(path)
    print(f"Изображение сохранено: {path}")

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

def phase_to_rgb(phase):
    """Преобразование фазы [-π, π] в RGB цветовое пространство."""
    phase_norm = (phase + np.pi) / (2 * np.pi)  # в диапазон [0, 1]
    phase_norm = np.clip(phase_norm, 0, 1)  # На всякий случай ограничиваем
    
    hsv = np.zeros((*phase.shape, 3))
    hsv[..., 0] = phase_norm  # Hue
    hsv[..., 1] = 1.0  # Saturation
    hsv[..., 2] = 1.0  # Value
    
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(hsv)

def get_transform_parameters():
    """Получение параметров преобразования от пользователя."""
    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ ПРЕОБРАЗОВАНИЯ")
    print("=" * 60)
    print("Выберите способ ввода параметра преобразования:")
    print("[1] Ввести угол α (в радианах или с pi)")
    print("[2] Ввести параметр H = tan(α)")
    
    input_type = input("Ваш выбор (1 или 2): ").strip()
    
    if input_type == '1':
        print("\nВведите угол α:")
        print("Примеры: '0.25' → 0.25π, 'pi/4' → π/4, '0.5*pi' → 0.5π, '45deg' → 45°")
        print("Для Фурье-преобразования: 'pi/2' или '90deg'")
        print("Для тождественного преобразования: '0' или '0deg'")
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
            return None, None
            
    elif input_type == '2':
        print("\nВведите параметр H:")
        print("Примеры: '0' → тождественное, '1' → α=π/4, 'inf' → α=π/2 (Фурье)")
        print("Также можно: 'tan(pi/4)' → 1, 'tan(45deg)' → 1")
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
            return None, None
    else:
        print("\nИспользуется значение по умолчанию: H = 0")
        alpha = 0.0
        H = 0.0
    
    return alpha, H

# ==== ОСНОВНАЯ ПРОГРАММА ====
def main():
    print("=" * 60)
    print("ГИРАТОРНОЕ ПРЕОБРАЗОВАНИЕ ЦВЕТНОГО ИЗОБРАЖЕНИЯ")
    print("=" * 60)
    
    # Получение параметров преобразования
    alpha, H = get_transform_parameters()
    if alpha is None or H is None:
        print("Не удалось получить параметры преобразования.")
        return
    
    # ==== ЗАГРУЗКА ИЗОБРАЖЕНИЯ ====
    print(f"\nЗагрузка изображения: {image_path}")
    image_rgb = load_color_image(image_path, size=resize_to)
    print(f"Размер изображения: {image_rgb.shape}")
    
    # ==== СОЗДАНИЕ ГИРАТОРА ====
    print(f"\nСоздание гиратора с размером {resize_to}x{resize_to}...")
    gyrator = Gyrator(size=resize_to, scale=1.0)
    
    print(f"\nПАРАМЕТРЫ ПРЕОБРАЗОВАНИЯ:")
    print(f"  Угол α: {alpha:.6f} рад ({alpha/np.pi:.6f}π)")
    print(f"  Параметр H: {H}")
    
    # ==== ОБРАБОТКА КАЖДОГО ЦВЕТОВОГО КАНАЛА ====
    magnitude_channels = []
    phase_channels = []
    
    for i, channel_name in enumerate(['R', 'G', 'B']):
        print(f'\nОбработка канала {channel_name}...')
        
        # Извлечение канала
        channel = image_rgb[:, :, i]
        
        # Подготовка входного поля (преобразование в комплексное)
        complex_field = channel.astype(np.complex128)
        
        # Применение гираторного преобразования
        if np.isinf(H):
            # Особые случаи для бесконечного H
            if H > 0:
                # Преобразование Фурье
                transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_field)))
            else:
                # Обратное преобразование Фурье
                transformed = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(complex_field)))
        else:
            # Обычное гираторное преобразование
            transformed = gyrator.transform(complex_field, alpha)
        
        # Сохранение результатов
        magnitude_channels.append(np.abs(transformed))
        phase_channels.append(np.angle(transformed))
    
    # ==== СОЗДАНИЕ RGB ИЗОБРАЖЕНИЙ ====
    # Модуль - нормализуем каждый канал отдельно
    mag_normalized = []
    for i in range(3):
        mag_norm = normalize_image(magnitude_channels[i])
        mag_normalized.append(mag_norm)
    
    mag_rgb = np.stack(mag_normalized, axis=2)
    
    # Фаза (усреднение фазовых карт по каналам)
    phase_combined = np.mean(phase_channels, axis=0)
    phase_rgb = phase_to_rgb(phase_combined)
    
    # ==== ВИЗУАЛИЗАЦИЯ ====
    visualize_color_image(image_rgb, f"Исходное изображение ({resize_to}x{resize_to})")
    visualize_color_image(mag_rgb, f"Модуль гираторного преобразования (α={alpha/np.pi:.3f}π)")
    visualize_color_image(phase_rgb, f"Фаза гираторного преобразования (α={alpha/np.pi:.3f}π)")
    
    # ==== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====
    save_color_image(mag_rgb, output_magnitude_path)
    save_color_image(phase_rgb, output_phase_path)
    
    print(f"\nРезультаты сохранены:")
    print(f"  Модуль: {os.path.abspath(output_magnitude_path)}")
    print(f"  Фаза:   {os.path.abspath(output_phase_path)}")
    
    # ==== ПРОВЕРКА УНИТАРНОСТИ ====
    print(f"\nПРОВЕРКА УНИТАРНОСТИ:")
    for i, channel_name in enumerate(['R', 'G', 'B']):
        energy_original = np.sum(np.abs(image_rgb[:, :, i])**2)
        energy_transformed = np.sum(magnitude_channels[i]**2)
        energy_ratio = energy_transformed / energy_original if energy_original > 0 else 1.0
        print(f"  Канал {channel_name}: энергия сохранена с точностью {abs(energy_ratio-1.0):.2e}")

if __name__ == "__main__":
    main()