"""
Гираторное преобразование для оптики.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import math


class Gyrator:
    """Класс для выполнения гираторного преобразования с сохранением энергии."""

    def __init__(self, size=256, scale=1.0):
        """
        Инициализация гиратора.

        Args:
            size: Размер сетки (квадратная)
            scale: Масштаб координатной сетки
        """
        self.size = size
        self.scale = scale
        
        # Правильное создание координатных сеток
        dx = 2.0 * scale / size
        dy = 2.0 * scale / size
        
        # Координаты центров пикселей
        x = np.linspace(-scale + dx/2, scale - dx/2, size)
        y = np.linspace(-scale + dy/2, scale - dy/2, size)
        
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.dx = dx
        self.dy = dy
        
        # Нормализация для сохранения энергии
        self.norm_factor = np.sqrt(self.dx * self.dy)
        
        # Частотные сетки
        fx = np.fft.fftfreq(size, d=dx)
        fy = np.fft.fftfreq(size, d=dy)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing='ij')
        
        # Частотная сетка с центрированием
        self.FX_shifted, self.FY_shifted = np.meshgrid(
            np.fft.fftshift(fx), 
            np.fft.fftshift(fy),
            indexing='ij'
        )

    def transform(self, field, alpha):
        """
        Выполняет гираторное преобразование.

        Args:
            field: 2D комплексное поле (с нулевой частотой в центре)
            alpha: Угол поворота в радианах

        Returns:
            Преобразованное поле (с нулевой частотой в центре)
        """
        if field.shape != (self.size, self.size):
            raise ValueError(f"Размер поля {field.shape} не соответствует размеру гиратора {self.size}x{self.size}")

        # Нормализуем входное поле для сохранения энергии
        field_normalized = field * self.norm_factor
        
        # Особые случаи
        if abs(np.sin(alpha)) < 1e-10:
            if abs(alpha) < 1e-10 or abs(alpha - 2*np.pi) < 1e-10:
                return field
            elif abs(alpha - np.pi) < 1e-10 or abs(alpha + np.pi) < 1e-10:
                return np.flipud(np.fliplr(field))
        
        return self._transform_fast(field_normalized, alpha)

    def _transform_fast(self, field, alpha):
        """
        Быстрый метод через два БПФ и три чирп-умножения.
        Алгоритм из статьи: Liu et al., "Optical image encryption via the gyrator transform", 2008.
        """
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        
        if abs(sin_alpha) < 1e-10:
            return field
        
        # Шаг 1: Первое чирп-умножение
        phase1 = np.pi * cos_alpha / sin_alpha * (self.X**2 + self.Y**2)
        chirp1 = np.exp(1j * phase1)
        field1 = field * chirp1
        
        # Шаг 2: Первое преобразование Фурье (БПФ)
        field1_fft = fftshift(field1)  # Центрируем для БПФ
        field1_fft = fft2(field1_fft)
        field1_fft = fftshift(field1_fft)  # Возвращаем в центрированный вид
        
        # Шаг 3: Второе чирп-умножение в частотной области
        # Масштабируем частотные координаты
        FX_scaled = self.FX_shifted / sin_alpha
        FY_scaled = self.FY_shifted / sin_alpha
        
        phase2 = np.pi * cos_alpha * sin_alpha * (FX_scaled**2 + FY_scaled**2)
        chirp2 = np.exp(1j * phase2)
        field2_fft = field1_fft * chirp2
        
        # Шаг 4: Второе преобразование Фурье
        field2 = ifftshift(field2_fft)  # Для обратного БПФ
        field2 = ifft2(field2)
        field2 = fftshift(field2)  # Центрируем
        
        # Шаг 5: Третье чирп-умножение
        phase3 = np.pi * cos_alpha / sin_alpha * (self.X**2 + self.Y**2)
        chirp3 = np.exp(1j * phase3)
        result = chirp3 * field2 / abs(sin_alpha)
        
        return result

    def inverse_transform(self, field, alpha):
        """
        Обратное гираторное преобразование.
        Для гиратора обратное преобразование соответствует углу -α.
        """
        return self.transform(field, -alpha)


# Вспомогательные функции для работы с изображениями
def prepare_input_image(image, expected_size=256):
    """
    Подготавливает изображение для гираторного преобразования.
    
    Args:
        image: Входное изображение (2D массив)
        expected_size: Ожидаемый размер
        
    Returns:
        Подготовленное комплексное поле
    """
    # Если изображение меньше ожидаемого размера, центрируем его
    if image.shape != (expected_size, expected_size):
        # Создаем новое изображение нужного размера
        new_image = np.zeros((expected_size, expected_size), dtype=complex)
        
        # Находим центр для вставки
        h, w = image.shape
        y_start = (expected_size - h) // 2
        x_start = (expected_size - w) // 2
        
        # Вставляем изображение в центр
        new_image[y_start:y_start+h, x_start:x_start+w] = image
        
        return new_image
    
    return image.astype(complex)


def show_gyrator_result(result):
    """
    Отображает результат гираторного преобразования.
    
    Args:
        result: Результат преобразования
    """
    import matplotlib.pyplot as plt
    
    # Создаем фигуру с двумя подграфиками
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Амплитуда
    im1 = axes[0].imshow(np.abs(result), cmap='hot')
    axes[0].set_title('Амплитуда')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Фаза
    im2 = axes[1].imshow(np.angle(result), cmap='hsv')
    axes[1].set_title('Фаза')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


# Функции парсинга остаются без изменений
def parse_angle_input(angle_str):
    """Парсит строку с углом."""
    import math
    
    if not angle_str:
        return 0.0
    
    angle_str = angle_str.strip().lower().replace(' ', '')
    
    # Обработка градусов
    if 'deg' in angle_str:
        deg_str = angle_str.replace('deg', '')
        try:
            deg = float(deg_str)
            return math.radians(deg)
        except ValueError:
            pass
    
    # Обработка радиан
    if 'rad' in angle_str:
        rad_str = angle_str.replace('rad', '')
        try:
            return float(rad_str)
        except ValueError:
            pass
    
    # Проверяем, содержит ли строка "pi"
    if 'pi' in angle_str:
        pi_factor_str = angle_str.replace('pi', '')
        
        if pi_factor_str == '' or pi_factor_str == '+':
            return math.pi
        elif pi_factor_str == '-':
            return -math.pi
        
        if '/' in pi_factor_str:
            parts = pi_factor_str.split('/')
            if parts[0] == '':
                numerator = 1
            elif parts[0] == '-':
                numerator = -1
            else:
                numerator = float(parts[0])
            denominator = float(parts[1])
            return numerator / denominator * math.pi
        elif '*' in pi_factor_str:
            parts = pi_factor_str.split('*')
            if parts[0] == '':
                factor = float(parts[1]) if len(parts) > 1 else 1
            else:
                factor = float(parts[0])
            return factor * math.pi
        else:
            try:
                return float(pi_factor_str) * math.pi
            except ValueError:
                if pi_factor_str.startswith('/'):
                    denominator = float(pi_factor_str[1:])
                    return math.pi / denominator
                else:
                    raise ValueError(f"Непонятный формат: {angle_str}")
    else:
        try:
            value = float(angle_str)
            return value * math.pi
        except ValueError:
            try:
                expr_str = angle_str.replace('^', '**')
                value = eval(expr_str)
                return value * math.pi
            except:
                raise ValueError(f"Невозможно преобразовать в число: {angle_str}")


def parse_H_input(H_str):
    """Парсит строку с параметром H."""
    import math
    
    if not H_str:
        return 0.0
    
    H_str = H_str.strip().lower().replace(' ', '')
    
    if H_str in ['inf', '?', '+inf', '+?']:
        return float('inf')
    elif H_str in ['-inf', '-?']:
        return float('-inf')
    
    if 'tan(' in H_str or 'arctan(' in H_str or 'atan(' in H_str:
        if 'tan(' in H_str:
            arg_str = H_str[4:-1]
            try:
                arg = parse_angle_input(arg_str)
                return math.tan(arg)
            except:
                try:
                    arg = float(arg_str)
                    return math.tan(arg)
                except:
                    raise ValueError(f"Не могу разобрать аргумент: {arg_str}")
        else:
            arg_str = H_str[7:-1] if 'arctan(' in H_str else H_str[4:-1]
            try:
                arg = float(arg_str)
                return math.atan(arg)
            except:
                raise ValueError(f"Не могу разобрать аргумент: {arg_str}")
    
    try:
        return float(H_str)
    except ValueError:
        raise ValueError(f"Не могу разобрать H: {H_str}")


def create_gyrator(size=256, scale=1.0):
    """Фабричная функция для создания гиратора."""
    return Gyrator(size, scale)
