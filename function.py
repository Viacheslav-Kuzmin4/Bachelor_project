import numpy as np
import matplotlib.pyplot as plt
from gyrator import Gyrator, create_gyrator, parse_angle_input, parse_H_input


# ============================================================================
# ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ РАЗЛИЧНЫХ ПУЧКОВ
# ============================================================================

def gaussian_beam_with_cubic_phase(size, w0, a, b, center=None):
    """
    Генерирует 2D гауссов пучок с кубической фазой.
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Гауссова амплитуда
    r_sq = x**2 + y**2
    gaussian = np.exp(-r_sq / (w0**2))
    
    # Кубическая фаза
    cubic_phase = np.exp(1j * (a * x**3 + b * y**3))
    
    return gaussian * cubic_phase


def gaussian_beam_with_quadratic_phase(size, w0, c, d, center=None):
    """
    Генерирует 2D гауссов пучок с квадратичной фазой (линза).
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Гауссова амплитуда
    r_sq = x**2 + y**2
    gaussian = np.exp(-r_sq / (w0**2))
    
    # Квадратичная фаза (линза)
    quadratic_phase = np.exp(1j * (c * x**2 + d * y**2))
    
    return gaussian * quadratic_phase


def simple_gaussian_beam(size, w0, center=None):
    """
    Генерирует простой 2D гауссов пучок без дополнительной фазы.
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Гауссова амплитуда
    r_sq = x**2 + y**2
    gaussian = np.exp(-r_sq / (w0**2))
    
    return gaussian


def plane_wave_with_linear_phase(size, kx, ky, center=None):
    """
    Генерирует плоскую волну с линейной фазой.
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Линейная фаза (плоская волна)
    linear_phase = np.exp(1j * (kx * x + ky * y))
    
    return linear_phase


def hyperbolic_phase_wave(size, c, center=None):
    """
    Генерирует волну с гиперболической фазой: exp(i*2π*c*x*y)
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Гиперболическая фаза
    hyperbolic_phase = np.exp(1j * 2 * np.pi * c * x * y)
    
    return hyperbolic_phase


def spherical_wave(size, b, center=None):
    """
    Генерирует сферическую волну: exp(-iπ*b*(x^2+y^2))
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Сферическая фаза
    r_sq = x**2 + y**2
    spherical_phase = np.exp(-1j * np.pi * b * r_sq)
    
    return spherical_phase


def hermit_gaussian_mode(size, m, n, w0=30, center=None):
    """
    Генерирует моды Эрмита-Гаусса HG_{m,n}.
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Гауссова огибающая
    r_sq = x**2 + y**2
    gaussian = np.exp(-r_sq / (w0**2))
    
    # Полиномы Эрмита
    from scipy.special import hermite
    
    Hm = hermite(m)
    Hn = hermite(n)
    
    # Нормировочный коэффициент
    norm = 1.0 / np.sqrt(2**m * np.math.factorial(m) * 2**n * np.math.factorial(n) * np.pi)
    
    # Мода Эрмита-Гаусса
    hermite_part = Hm(np.sqrt(2) * x / w0) * Hn(np.sqrt(2) * y / w0)
    
    return norm * hermite_part * gaussian


def circle_function(size, radius, center=None):
    """
    Генерирует круговую функцию (цилиндрическую).
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.arange(size) - center[0]
    y = np.arange(size) - center[1]
    x, y = np.meshgrid(x, y)
    
    # Круговая функция
    r_sq = x**2 + y**2
    circle = np.zeros_like(r_sq, dtype=complex)
    circle[r_sq <= radius**2] = 1.0 + 0j
    
    return circle


# ============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================================================

def visualize_beam(beam, title, is_complex=True):
    """
    Отображает 2D пучок.
    """
    plt.figure(figsize=(10, 8))
    
    if is_complex:
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(beam), cmap='gray')
        plt.title(f"{title} (Амплитуда)")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(beam), cmap='gray', vmin=-np.pi, vmax=np.pi)
        plt.title(f"{title} (Фаза)")
        plt.colorbar()
    else:
        plt.imshow(beam, cmap='gray')
        plt.title(title)
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def visualize_spectral_coordinate_domain(field, title, alpha):
    """
    Визуализирует поле в спектрально-координатной области.
    Показывает фазу в координатном и спектральном (частотном) пространстве.
    """
    # Вычисляем 2D Фурье-образ (спектр)
    spectrum = np.fft.fft2(field)
    spectrum = np.fft.fftshift(spectrum)  # Центрируем нулевую частоту
    
    # Создаем фигуру с двумя подграфиками
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Фаза в координатном пространстве
    im1 = axes[0].imshow(np.angle(field), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(f"{title}\nФаза в координатном пространстве\nα={alpha/np.pi:.3f}π")
    axes[0].set_xlabel('x (пиксели)')
    axes[0].set_ylabel('y (пиксели)')
    plt.colorbar(im1, ax=axes[0])
    
    # Фаза в спектральном (частотном) пространстве
    im2 = axes[1].imshow(np.angle(spectrum), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"{title}\nФаза в спектральном пространстве\nα={alpha/np.pi:.3f}π")
    axes[1].set_xlabel('Частота по x')
    axes[1].set_ylabel('Частота по y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def visualize_comparison(original, transformed, alpha, params_str=""):
    
    # Преобразованный пучок
    ax3 = plt.subplot(1, 2, 1)
    im3 = ax3.imshow(np.abs(transformed), cmap='gray')
    ax3.set_title(f"Гираторное преобразование (Амплитуда)\nα={alpha/np.pi:.3f}π")
    plt.colorbar(im3, ax=ax3)
    
    ax4 = plt.subplot(1, 2, 2)
    im4 = ax4.imshow(np.angle(transformed), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax4.set_title(f"Гираторное преобразование (Фаза)\nα={alpha/np.pi:.3f}π")
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.show()


def check_unitarity(original, transformed):
    """
    Проверка унитарности преобразования.
    """
    energy_original = np.sum(np.abs(original)**2)
    energy_transformed = np.sum(np.abs(transformed)**2)
    
    # Избегаем деления на ноль
    if energy_original > 0:
        energy_ratio = energy_transformed / energy_original
    else:
        energy_ratio = 1.0  # Если исходная энергия 0, то и преобразованная должна быть 0
    
    print(f"\nПроверка унитарности:")
    print(f"  Энергия исходного: {energy_original:.6e}")
    print(f"  Энергия преобразованного: {energy_transformed:.6e}")
    print(f"  Отношение энергий: {energy_ratio:.6f}")
    
    if abs(energy_ratio - 1.0) < 1e-0:
        print("  ✓ Преобразование унитарно (сохраняет энергию)")
        return True
    else:
        print(f"  ⚠ Преобразование не унитарно (отклонение: {abs(energy_ratio-1.0):.2e})")
        return False


def check_reversibility(original, gyrator, alpha):
    """
    Проверка обратимости гираторного преобразования.
    """
    print(f"\nПроверка обратимости для α={alpha/np.pi:.3f}π:")
    
    # Прямое преобразование
    transformed = gyrator.transform(original, alpha)
    
    # Обратное преобразование
    restored = gyrator.inverse_transform(transformed, alpha)
    
    # Вычисление ошибок
    abs_error = np.mean(np.abs(original - restored))
    # Относительная ошибка (избегаем деления на ноль)
    nonzero_mask = np.abs(original) > 1e-10
    if np.any(nonzero_mask):
        rel_error = np.mean(np.abs(original[nonzero_mask] - restored[nonzero_mask]) / np.abs(original[nonzero_mask]))
    else:
        rel_error = abs_error
    max_error = np.max(np.abs(original - restored))
    
    print(f"  Средняя абсолютная ошибка: {abs_error:.6e}")
    print(f"  Средняя относительная ошибка: {rel_error:.6e}")
    print(f"  Максимальная ошибка: {max_error:.6e}")
    
    if abs_error < 1e-6:
        print("  ✓ Преобразование обратимо с высокой точностью")
        return True, abs_error
    elif abs_error < 1e-0:
        print("  ✓ Преобразование обратимо с удовлетворительной точностью")
        return True, abs_error
    else:
        print("  ⚠ Преобразование не полностью обратимо")
        return False, abs_error

# ============================================================================
# ФУНКЦИИ ВЫБОРА И КОНФИГУРАЦИИ
# ============================================================================

def select_function(function_number):
    """
    Функция для выбора типа функции.
    """
    print(f"\n{'='*60}")
    print(f"ВЫБОР ФУНКЦИИ #{function_number}")
    print("="*60)
    print("Доступные функции:")
    print("   [1] Гауссов пучок с кубической фазой")
    print("   [2] Гауссов пучок с квадратичной фазой (линза)")
    print("   [3] Простой гауссов пучок (без дополнительной фазы)")
    print("   [4] Плоская волна с линейной фазой")
    print("   [5] Волна с гиперболической фазой")
    print("   [6] Сферическая волна")
    print("   [7] Мода Эрмита-Гаусса")
    print("   [8] Круговая функция (цилиндрическая)")
    
    choice = input("   Ваш выбор (1-8): ").strip()
    
    return choice


def configure_function(choice, size, w0_default=30):
    """
    Конфигурирует выбранную функцию, запрашивая параметры.
    Возвращает функцию и строку с параметрами.
    """
    # Общие параметры
    center = (size // 2, size // 2)
    
    if choice == '1':  # Гауссов пучок с кубической фазой
        print("\nГауссов пучок с кубической фазой")
        w0 = float(input(f"   Радиус перетяжки w0 (по умолчанию {w0_default}): ") or w0_default)
        a = float(input("   Коэффициент a для x^3 (по умолчанию 0.0005): ") or "0.0005")
        b = float(input("   Коэффициент b для y^3 (по умолчанию 0.0005): ") or "0.0005")
        
        def func():
            return gaussian_beam_with_cubic_phase(size, w0, a, b, center)
        
        params_str = f"Гауссов пучок с кубической фазой: w0={w0}, a={a}, b={b}"
        
    elif choice == '2':  # Гауссов пучок с квадратичной фазой
        print("\nГауссов пучок с квадратичной фазой (линза)")
        w0 = float(input(f"   Радиус перетяжки w0 (по умолчанию {w0_default}): ") or w0_default)
        c = float(input("   Коэффициент c для x^2 (по умолчанию 0.001): ") or "0.001")
        d = float(input("   Коэффициент d для y^2 (по умолчанию 0.001): ") or "0.001")
        
        def func():
            return gaussian_beam_with_quadratic_phase(size, w0, c, d, center)
        
        params_str = f"Гауссов пучок с квадратичной фазой: w0={w0}, c={c}, d={d}"
        
    elif choice == '3':  # Простой гауссов пучок
        print("\nПростой гауссов пучок (без дополнительной фазы)")
        w0 = float(input(f"   Радиус перетяжки w0 (по умолчанию {w0_default}): ") or w0_default)
        
        def func():
            return simple_gaussian_beam(size, w0, center)
        
        params_str = f"Простой гауссов пучок: w0={w0}"
        
    elif choice == '4':  # Плоская волна с линейной фазой
        print("\nПлоская волна с линейной фазой")
        kx = float(input("   Волновое число по x (по умолчанию 0.1): ") or "0.1")
        ky = float(input("   Волновое число по y (по умолчанию 0.1): ") or "0.1")
        
        def func():
            return plane_wave_with_linear_phase(size, kx, ky, center)
        
        params_str = f"Плоская волна с линейной фазой: kx={kx}, ky={ky}"
        
    elif choice == '5':  # Волна с гиперболической фазой
        print("\nВолна с гиперболической фазой: exp(i*2π*c*x*y)")
        c = float(input("   Коэффициент c (по умолчанию 0.01): ") or "0.01")
        
        def func():
            return hyperbolic_phase_wave(size, c, center)
        
        params_str = f"Волна с гиперболической фазой: c={c}"
        
    elif choice == '6':  # Сферическая волна
        print("\nСферическая волна: exp(-iπ*b*(x^2+y^2))")
        b = float(input("   Коэффициент b (по умолчанию 0.001): ") or "0.001")
        
        def func():
            return spherical_wave(size, b, center)
        
        params_str = f"Сферическая волна: b={b}"
        
    elif choice == '7':  # Мода Эрмита-Гаусса
        print("\nМода Эрмита-Гаусса HG_{m,n}")
        m = int(input("   Индекс m (по умолчанию 2): ") or "2")
        n = int(input("   Индекс n (по умолчанию 1): ") or "1")
        w0 = float(input(f"   Радиус перетяжки w0 (по умолчанию {w0_default}): ") or w0_default)
        
        def func():
            return hermit_gaussian_mode(size, m, n, w0, center)
        
        params_str = f"Мода Эрмита-Гаусса: HG_{{{m},{n}}}, w0={w0}"
        
    elif choice == '8':  # Круговая функция
        print("\nКруговая функция (цилиндрическая)")
        radius = float(input("   Радиус круга (по умолчанию 15): ") or "15")
        
        def func():
            return circle_function(size, radius, center)
        
        params_str = f"Круговая функция: радиус={radius}"
        
    else:
        print("   Неверный выбор, используется гауссов пучок с кубической фазой по умолчанию")
        w0 = w0_default
        a = 0.0005
        b = 0.0005
        
        def func():
            return gaussian_beam_with_cubic_phase(size, w0, a, b, center)
        
        params_str = f"Гауссов пучок с кубической фазой (по умолчанию): w0={w0}, a={a}, b={b}"
    
    return func, params_str


def select_visualization_type():
    """
    Выбор типа визуализации для преобразованной функции.
    """
    print("\nВыбор типа визуализации:")
    print("   [1] Обычная визуализация (амплитуда и фаза)")
    print("   [2] Спектрально-координатная область (фаза в координатном и спектральном пространстве)")
    
    choice = input("   Ваш выбор (1 или 2): ").strip()
    
    if choice == '1':
        return ['standard']
    elif choice == '2':
        return ['spectral']
    else:
        print("   Неверный выбор, используется обычная визуализация")
        return ['standard']


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Основная функция программы.
    """
    print("=" * 80)
    print("ГИРАТОРНОЕ ПРЕОБРАЗОВАНИЕ ДЛЯ РАЗЛИЧНЫХ ФУНКЦИЙ")
    print("=" * 80)
    
    # Параметры по умолчанию
    size = 64
    w0_default = 30
    
    # Создание гиратора
    print("\n СОЗДАНИЕ ГИРАТОРА")
    print(f"   Размер сетки: {size}x{size}")
    gyrator = create_gyrator(size=size, scale=2.0)
    print("  Гиратор создан")

    # Выбор количества функций
    print("\n   Сколько функций вы хотите преобразовать?")
    num_functions = int(input("   Введите число: ") or "1")
    
    # Выбор и конфигурация функций
    functions = []
    func_params = []
    vis_types = []
    angles = []
    tangens = []
    for i in range(num_functions):
        func_choice = select_function(i+1)
        func, params_str = configure_function(func_choice, size, w0_default)
        functions.append(func)
        func_params.append(params_str)
        
        # Выбор типа визуализации
        vis_types.append(select_visualization_type())
        print("\n ПАРАМЕТРЫ ПРЕОБРАЗОВАНИЯ")
        print("   [1] Ввести угол α (в радианах или с pi)")
        print("   [2] Ввести параметр H = tan(α)")
        
        input_type = input("   Ваш выбор (1 или 2): ").strip()
        
        if input_type == '1':
            print("\n   Введите угол α:")
            print("   Примеры: '0.25' → 0.25π, 'pi/4' → π/4, '0.5*pi' → 0.5π, '45deg' → 45°")
            print("   Для Фурье-преобразования: '0.5' (половина π), '1' (полный π)")
            print("   Для тождественного преобразования: '0' или '0deg'")
            alpha_str = input("   α = ").strip()
            
            try:
                alpha = parse_angle_input(alpha_str)
                H = np.tan(alpha)
                print(f"   α = {alpha:.6f} рад = {alpha/np.pi:.3f}π")
                print(f"   H = tan(α) = {H:.6f}")
                angles.append(alpha)
                tangens.append(H)
            except Exception as e:
                print(f"   Ошибка: {e}")
                return
                
        elif input_type == '2':
            print("\n   Введите параметр H:")
            print("   Примеры: '0' → тождественное, '1' → α=π/4, 'inf' → α=π/2 (Фурье)")
            print("   Также можно: 'tan(pi/4)' → 1, 'tan(45deg)' → 1")
            H_str = input("   H = ").strip()
            
            try:
                H = parse_H_input(H_str)
                if np.isinf(H):
                    alpha = np.pi/2 * np.sign(H)
                elif H == 0:
                    alpha = 0.0
                else:
                    alpha = np.arctan(H)
                print(f"   H = {H}")
                print(f"   α = arctan(H) = {alpha:.6f} рад = {alpha/np.pi:.3f}π")
                angles.append(alpha)
                tangens.append(H)
            except Exception as e:
                print(f"   Ошибка: {e}")
                return
        else:
            print("\n   Используется значение по умолчанию: H = 0")
            angles.append(0.0)
            tangens.append(0.0)
    
    print(f"   Количество функций: {len(functions)}")
    for i, params in enumerate(func_params):
        print(f"\n   Функция #{i+1}:")
        print(f"     {params}")
        print(f"     Тип визуализации: {', '.join(vis_types[i])}")
    
    print("=" * 80)
    
    try:
        # Выполнение преобразования для каждой функции
        for i, (func, params_str, vis_type, alpha, H) in enumerate(zip(
            functions, func_params, vis_types, angles, tangens)):
            
            print(f"\n{'='*80}")
            print(f"ФУНКЦИЯ #{i+1}: {params_str}")
            print("="*80)
            print("\n" + "=" * 80)
            # Вывод сводки параметров
            print("СВОДКА ПАРАМЕТРОВ:")
            print("=" * 80)
            print(f"   Угол α: {alpha:.6f} рад ({alpha/np.pi:.3f}π)")
            print(f"   Параметр H: {H}")
            # Генерация пучка
            print("Генерация пучка...")
            beam = func()
            
            # Визуализация исходного пучка
            print("Визуализация исходного пучка...")
            visualize_beam(beam, f"Функция #{i+1}: {params_str}")
            
            # Гираторное преобразование
            print("Выполнение гираторного преобразования...")
            transformed = gyrator.transform(beam, alpha)
            
            # Проверка унитарности
            check_unitarity(beam, transformed)
            
            # Визуализация результатов
            print("\nВизуализация результатов...")
            
            # Выбор типа визуализации
            for vtype in vis_type:
                if vtype == 'standard':
                    visualize_comparison(beam, transformed, alpha, params_str)
                elif vtype == 'spectral':
                    visualize_spectral_coordinate_domain(
                        transformed, 
                        f"Функция #{i+1}: {params_str}", 
                        alpha
                    )
            
            # Проверка обратимости (кроме особых случаев)
            if not (abs(alpha) < 1e-10 or abs(alpha - np.pi) < 1e-10):
                is_reversible, error = check_reversibility(beam, gyrator, alpha)
        
        print("\n" + "=" * 80)
        print("ПРЕОБРАЗОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()