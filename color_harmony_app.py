import sys
import math
import locale
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFrame, QTabWidget,
                           QScrollArea, QSizePolicy, QMenu, QColorDialog, QFileDialog, QMessageBox, QInputDialog)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal
from functools import lru_cache

# Флаги доступности библиотек
has_pil = False
has_numpy = False
has_sklearn = False

# Импортируем необязательные библиотеки
try:
    from PIL import Image
    has_pil = True
except ImportError:
    pass

try:
    import numpy as np
    has_numpy = True
except ImportError:
    pass

try:
    from sklearn.cluster import KMeans
    has_sklearn = True
except ImportError:
    pass

import os
import os.path

# Словарь с переводами для разных языков
TRANSLATIONS = {
    "ru": {
        # Основные элементы интерфейса
        "app_title": "Анализатор гармонии цветов",
        "similarity_tab": "Сходство",
        "harmony_tab": "Гармония",
        "schemes_tab": "Цветовые схемы",
        
        # Вкладки анализа
        "similarity_title": "Анализ сходства цветов",
        "similarity_desc": "Эта функция поможет определить степень сходства между цветами.\nЧем выше процент, тем более похожи цвета друг на друга.",
        "harmony_title": "Анализ гармонии цветов",
        "harmony_desc": "Эта функция оценивает гармоничность сочетания цветов.\nУчитываются различные типы цветовых гармоний: монохроматическая, аналоговая, дополнительная и другие.",
        
        # Кнопки выбора цвета
        "color_1": "Цвет 1",
        "color_2": "Цвет 2",
        "base_color": "Базовый цвет",
        "select_colors": "Выберите два цвета для анализа",
        
        # Информация о цвете
        "color_info": "Информация о цвете",
        "color_number": "Цвет {0}",
        
        # Результаты
        "similarity_result": "Сходство цветов: {0}%",
        "harmony_type": "Тип гармонии: {0}",
        "harmony_score": "Оценка: {0}% ({1})",
        "excellent": "Отлично",
        "good": "Хорошо",
        "satisfactory": "Удовлетворительно",
        "poor": "Плохо",
        
        # Вкладка схем
        "schemes_title": "Генератор цветовых схем",
        "schemes_desc": "Выберите базовый цвет, и программа создаст различные гармоничные цветовые схемы на его основе.\nКаждая схема демонстрирует определённый тип цветовой гармонии.",
        "extract_from_image": "Извлечь цвета из изображения",
        
        # Опции экспорта
        "export_to_css": "Экспортировать в CSS",
        "export_to_scss": "Экспортировать в SCSS",
        "copy_rgb": "Копировать RGB: {0}, {1}, {2}",
        "copy_hex": "Копировать HEX: {0}",
        
        # Сообщения диалогов
        "save_css": "Сохранить CSS файл",
        "save_scss": "Сохранить SCSS файл",
        "export_success": "Экспорт успешен",
        "css_saved": "CSS файл сохранен в: {0}",
        "scss_saved": "SCSS файл сохранен в: {0}",
        "export_error": "Ошибка экспорта",
        "file_save_error": "Не удалось сохранить файл: {0}",
        "missing_dependencies": "Отсутствуют зависимости",
        "dependencies_msg": "Для извлечения цветов необходимы библиотеки:\nPillow, NumPy, scikit-learn\n\nУстановите их с помощью pip:\npip install pillow numpy scikit-learn",
        "select_image": "Выбрать изображение",
        "colors_count": "Количество цветов",
        "colors_count_msg": "Укажите количество цветов для извлечения:",
        "image_error": "Ошибка при обработке изображения",
        "colors_from": "Цвета из {0}",
        
        # Типы гармонии
        "monochromatic": "Монохроматическая",
        "analogous": "Аналоговая",
        "complementary": "Дополнительная",
        "triadic": "Триадная",
        "rectangular": "Прямоугольная",
        "split_complementary": "Расщепленная комплементарная",
        "tetradic": "Тетрадная",
        "split_complementary2": "Комплементарная разделенная",
        "disharmony": "Дисгармония",
        
        # Градиентные схемы
        "light_gradient": "Градиент к светлому",
        "dark_gradient": "Градиент к тёмному",
        "complementary_gradient": "Градиент к дополнительному",
        "double_gradient": "Двойной градиент",
        "rainbow_gradient": "Радужный градиент",
        
        # Выбор языка
        "language": "Language",
    },
    
    "en": {
        # Основные элементы интерфейса
        "app_title": "Color Harmony Analyzer",
        "similarity_tab": "Similarity", 
        "harmony_tab": "Harmony",
        "schemes_tab": "Color Schemes",
        
        # Вкладки анализа
        "similarity_title": "Color Similarity Analysis",
        "similarity_desc": "This function helps determine the similarity degree between colors.\nThe higher the percentage, the more similar the colors are to each other.",
        "harmony_title": "Color Harmony Analysis",
        "harmony_desc": "This function evaluates the harmony of color combinations.\nVarious types of color harmonies are considered: monochromatic, analogous, complementary, and others.",
        
        # Кнопки выбора цвета
        "color_1": "Color 1",
        "color_2": "Color 2",
        "base_color": "Base Color",
        "select_colors": "Select two colors for analysis",
        
        # Информация о цвете
        "color_info": "Color Information",
        "color_number": "Color {0}",
        
        # Результаты
        "similarity_result": "Color similarity: {0}%",
        "harmony_type": "Harmony type: {0}",
        "harmony_score": "Score: {0}% ({1})",
        "excellent": "Excellent",
        "good": "Good", 
        "satisfactory": "Satisfactory",
        "poor": "Poor",
        
        # Вкладка схем
        "schemes_title": "Color Schemes Generator",
        "schemes_desc": "Select a base color, and the program will create various harmonious color schemes based on it.\nEach scheme demonstrates a specific type of color harmony.",
        "extract_from_image": "Extract colors from image",
        
        # Опции экспорта
        "export_to_css": "Export to CSS",
        "export_to_scss": "Export to SCSS", 
        "copy_rgb": "Copy RGB: {0}, {1}, {2}",
        "copy_hex": "Copy HEX: {0}",
        
        # Сообщения диалогов
        "save_css": "Save CSS file",
        "save_scss": "Save SCSS file", 
        "export_success": "Export Successful",
        "css_saved": "CSS file saved to: {0}",
        "scss_saved": "SCSS file saved to: {0}",
        "export_error": "Export Error",
        "file_save_error": "Failed to save file: {0}",
        "missing_dependencies": "Missing Dependencies",
        "dependencies_msg": "The following libraries are required for color extraction:\nPillow, NumPy, scikit-learn\n\nInstall them using pip:\npip install pillow numpy scikit-learn",
        "select_image": "Select Image",
        "colors_count": "Number of Colors",
        "colors_count_msg": "Specify the number of colors to extract:",
        "image_error": "Image Processing Error",
        "colors_from": "Colors from {0}",
        
        # Типы гармонии
        "monochromatic": "Monochromatic",
        "analogous": "Analogous",
        "complementary": "Complementary",
        "triadic": "Triadic",
        "rectangular": "Rectangular",
        "split_complementary": "Split Complementary",
        "tetradic": "Tetradic",
        "split_complementary2": "Split Complementary Variant",
        "disharmony": "Disharmony",
        
        # Градиентные схемы
        "light_gradient": "Gradient to Light",
        "dark_gradient": "Gradient to Dark",
        "complementary_gradient": "Gradient to Complementary",
        "double_gradient": "Double Gradient",
        "rainbow_gradient": "Rainbow Gradient",
        
        # Выбор языка
        "language": "Language",
    },
    
    "zh": {
        # Основные элементы интерфейса
        "app_title": "色彩和谐分析器",
        "similarity_tab": "相似度",
        "harmony_tab": "和谐性", 
        "schemes_tab": "配色方案",
        
        # Вкладки анализа
        "similarity_title": "色彩相似度分析",
        "similarity_desc": "此功能帮助确定颜色之间的相似程度。\n百分比越高，颜色之间越相似。",
        "harmony_title": "色彩和谐性分析",
        "harmony_desc": "此功能评估颜色组合的和谐性。\n考虑多种和谐类型：单色、类似色、互补色等。",
        
        # Кнопки выбора цвета
        "color_1": "颜色 1",
        "color_2": "颜色 2",
        "base_color": "基础颜色",
        "select_colors": "选择两种颜色进行分析",
        
        # Информация о цвете
        "color_info": "颜色信息",
        "color_number": "颜色 {0}",
        
        # Результаты
        "similarity_result": "颜色相似度: {0}%",
        "harmony_type": "和谐类型: {0}",
        "harmony_score": "评分: {0}% ({1})",
        "excellent": "优秀",
        "good": "良好",
        "satisfactory": "一般",
        "poor": "较差",
        
        # Вкладка схем
        "schemes_title": "配色方案生成器",
        "schemes_desc": "选择一种基础颜色，程序将创建各种和谐的配色方案。\n每种方案展示特定类型的色彩和谐。",
        "extract_from_image": "从图像提取颜色",
        
        # Опции экспорта
        "export_to_css": "导出为CSS",
        "export_to_scss": "导出为SCSS",
        "copy_rgb": "复制RGB: {0}, {1}, {2}",
        "copy_hex": "复制HEX: {0}",
        
        # Сообщения диалогов
        "save_css": "保存CSS文件",
        "save_scss": "保存SCSS文件",
        "export_success": "导出成功",
        "css_saved": "CSS文件已保存至: {0}",
        "scss_saved": "SCSS文件已保存至: {0}",
        "export_error": "导出错误",
        "file_save_error": "无法保存文件: {0}",
        "missing_dependencies": "缺少依赖库",
        "dependencies_msg": "提取颜色需要以下库:\nPillow, NumPy, scikit-learn\n\n使用pip安装:\npip install pillow numpy scikit-learn",
        "select_image": "选择图像",
        "colors_count": "颜色数量",
        "colors_count_msg": "指定要提取的颜色数量:",
        "image_error": "图像处理错误",
        "colors_from": "来自{0}的颜色",
        
        # Типы гармонии
        "monochromatic": "单色",
        "analogous": "类似色",
        "complementary": "互补色",
        "triadic": "三色",
        "rectangular": "矩形",
        "split_complementary": "分裂互补色",
        "tetradic": "四色",
        "split_complementary2": "分裂互补色变体",
        "disharmony": "不和谐",
        
        # Градиентные схемы
        "light_gradient": "浅色渐变",
        "dark_gradient": "深色渐变",
        "complementary_gradient": "互补色渐变",
        "double_gradient": "双向渐变",
        "rainbow_gradient": "彩虹渐变",
        
        # Выбор языка
        "language": "Language",
    }
}

class LanguageManager:
    """Управляет переводами и языковыми настройками приложения."""
    
    def __init__(self):
        # Русский язык по умолчанию
        self.current_language = "ru"
        self.translations = TRANSLATIONS
        self.detect_system_language()
    
    def detect_system_language(self):
        """Определяет системный язык и устанавливает его как текущий, если поддерживается."""
        try:
            # Получаем системную локаль с помощью рекомендуемых методов
            locale.setlocale(locale.LC_ALL, '')
            system_locale = locale.getlocale()[0]
            
            if system_locale:
                # Извлекаем код языка (первые 2 символа)
                lang_code = system_locale.split('_')[0].lower()
                # Если язык поддерживается, устанавливаем его как текущий
                if lang_code in self.translations:
                    self.current_language = lang_code
        except Exception:
            # В случае ошибки сохраняем язык по умолчанию
            pass
    
    def set_language(self, language_code):
        """Устанавливает текущий язык."""
        if language_code in self.translations:
            self.current_language = language_code
            return True
        return False
    
    def get_text(self, key, *args):
        """Получает переведенный текст для указанного ключа."""
        # Получаем перевод для текущего языка
        translation = self.translations.get(self.current_language, self.translations["en"])
        # Получаем текст для ключа, если не найден - используем английский
        text = translation.get(key, self.translations["en"].get(key, key))
        # Форматируем с аргументами, если они предоставлены
        if args:
            return text.format(*args)
        return text

# Создаем глобальный экземпляр менеджера языков
lang_manager = LanguageManager()

# Вспомогательная функция для удобного доступа к переводам
def tr(key, *args):
    """Вспомогательная функция для получения переведенного текста."""
    return lang_manager.get_text(key, *args)

def rgb_to_xyz(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Конвертирует RGB в XYZ цветовое пространство"""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Коррекция гаммы
    r = pow((r + 0.055) / 1.055, 2.4) if r > 0.04045 else r / 12.92
    g = pow((g + 0.055) / 1.055, 2.4) if g > 0.04045 else g / 12.92
    b = pow((b + 0.055) / 1.055, 2.4) if b > 0.04045 else b / 12.92
    
    # Преобразование в XYZ
    r *= 100
    g *= 100
    b *= 100
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    return x, y, z

def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Конвертирует XYZ в LAB цветовое пространство"""
    # Референсные значения для D65
    x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
    
    x, y, z = x / x_ref, y / y_ref, z / z_ref
    
    x = pow(x, 1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    y = pow(y, 1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    z = pow(z, 1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)
    
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    return L, a, b

def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Конвертирует RGB в LAB"""
    return xyz_to_lab(*rgb_to_xyz(r, g, b))

def calculate_delta_e_2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """
    Расчет дельта E по алгоритму CIEDE2000
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Среднее значение яркости
    avg_L = (L1 + L2) / 2
    
    # Вычисление параметров C (хроматичность)
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2
    
    # Поправка Gi
    G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    
    # Применение поправки к a
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # Вычисление C' и h'
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    avg_C_prime = (C1_prime + C2_prime) / 2
    
    h1_prime = math.atan2(b1, a1_prime) * 180 / math.pi
    if h1_prime < 0:
        h1_prime += 360
    
    h2_prime = math.atan2(b2, a2_prime) * 180 / math.pi
    if h2_prime < 0:
        h2_prime += 360
    
    # Вычисление разности hue
    if abs(h1_prime - h2_prime) > 180:
        if h2_prime <= h1_prime:
            h2_prime += 360
        else:
            h1_prime += 360
            
    delta_h_prime = h2_prime - h1_prime
    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_h_prime * math.pi / 360)
    
    # Среднее значение H'
    avg_H_prime = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > 180:
        avg_H_prime += 180
    
    # Вычисление весовых коэффициентов
    T = 1 - 0.17 * math.cos((avg_H_prime - 30) * math.pi / 180) + \
              0.24 * math.cos(2 * avg_H_prime * math.pi / 180) + \
              0.32 * math.cos((3 * avg_H_prime + 6) * math.pi / 180) - \
              0.20 * math.cos((4 * avg_H_prime - 63) * math.pi / 180)
    
    # Вычисление параметра SL
    SL = 1 + (0.015 * (avg_L - 50)**2) / math.sqrt(20 + (avg_L - 50)**2)
    SC = 1 + 0.045 * avg_C_prime
    SH = 1 + 0.015 * avg_C_prime * T
    
    # Поправки для вращения
    RC = 2 * math.sqrt(avg_C_prime**7 / (avg_C_prime**7 + 25**7))
    RT = -math.sin(60 * math.exp(-((avg_H_prime - 275) / 25)**2) * math.pi / 180) * RC
    
    # Вычисление дельта E по CIEDE2000
    kL, kC, kH = 1, 1, 1
    
    delta_L = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    delta_E = math.sqrt(
        (delta_L / (kL * SL))**2 + 
        (delta_C_prime / (kC * SC))**2 + 
        (delta_H_prime / (kH * SH))**2 + 
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    
    return delta_E

@lru_cache(maxsize=1000)
def calculate_delta_e_2000_cached(lab1, lab2):
    """Кэшированная версия расчета дельта E"""
    # Преобразуем кортежи в хешируемый формат
    lab1 = tuple(map(float, lab1))
    lab2 = tuple(map(float, lab2))
    return calculate_delta_e_2000(lab1, lab2)

def calculate_color_similarity(color1: tuple[int, int, int], color2: tuple[int, int, int]) -> float:
    """
    Рассчитывает сходство между двумя RGB цветами в процентах,
    используя алгоритм CIEDE2000 для более точного восприятия цвета.
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    # Преобразование в LAB
    lab1 = rgb_to_lab(r1, g1, b1)
    lab2 = rgb_to_lab(r2, g2, b2)
    
    # Расчет Delta E
    delta_e = calculate_delta_e_2000_cached(lab1, lab2)
    
    # Преобразование дельта E в проценты сходства
    # Дельта E = 0 означает идентичные цвета (100% сходство)
    # Используется экспоненциальное преобразование для более интуитивной шкалы
    # Значение 30 подобрано так, чтобы различимые цвета имели низкий процент сходства
    similarity = 100 * math.exp(-delta_e / 30)
    
    return round(similarity, 2)

def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Конвертирует RGB в HSV."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    
    # Защита от деления на очень маленькие числа
    EPSILON = 1e-10
    
    if abs(diff) < EPSILON:  # Используем epsilon вместо точного сравнения
        h = 0
    elif abs(cmax - r) < EPSILON:  # r является максимумом
        h = (60 * ((g - b) / max(diff, EPSILON)) + 360) % 360
    elif abs(cmax - g) < EPSILON:  # g является максимумом
        h = (60 * ((b - r) / max(diff, EPSILON)) + 120) % 360
    else:  # b является максимумом
        h = (60 * ((r - g) / max(diff, EPSILON)) + 240) % 360
    
    # Защита от деления на ноль при вычислении насыщенности
    s = 0 if abs(cmax) < EPSILON else (diff / max(cmax, EPSILON)) * 100
    v = cmax * 100
    return h, s, v

def calculate_harmony_score(color1: tuple[int, int, int], color2: tuple[int, int, int]) -> tuple[float, str]:
    """
    Оценивает гармоничность сочетания цветов по шкале от 0 до 100.
    Возвращает оценку и описание типа гармонии.
    """
    h1, s1, v1 = rgb_to_hsv(*color1)
    h2, s2, v2 = rgb_to_hsv(*color2)
    
    # Разница в оттенке (hue)
    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
    
    # Определяем тип гармонии с ключами для перевода
    harmony_types = {
        (0, 10): ("monochromatic", 90),        # Очень близкие оттенки
        (10, 30): ("analogous", 80),              # Близкие оттенки
        (150, 190): ("complementary", 85),        # Противоположные цвета
        (110, 130): ("triadic", 75),              # Треть цветового круга
        (85, 95): ("rectangular", 70),           # Четверть круга
        (30, 40): ("split_complementary", 78),
        (60, 70): ("tetradic", 72),
        (95, 105): ("split_complementary2", 75),
    }
    
    # Устанавливаем базовую оценку для дисгармонии
    harmony_score = 30  # Базовый счет для несоответствия известным типам гармонии
    harmony_name = "disharmony"
    
    for (min_angle, max_angle), (name, base_score) in harmony_types.items():
        if min_angle <= hue_diff <= max_angle:
            harmony_name = name
            harmony_score = base_score
            break
    
    # Корректируем оценку на основе насыщенности и яркости
    saturation_diff = abs(s1 - s2) / 100
    value_diff = abs(v1 - v2) / 100
    
    # Уменьшаем оценку, если есть большая разница в насыщенности или яркости
    harmony_score -= (saturation_diff * 20 + value_diff * 20)
    
    # Ограничиваем оценку диапазоном 0-100
    harmony_score = max(0, min(100, harmony_score))
    
    # Переводим название гармонии
    translated_name = tr(harmony_name)
    
    return round(harmony_score, 2), translated_name

def normalize_rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Нормализует RGB значения в диапазон 0-255."""
    return (
        max(0, min(255, round(r))),
        max(0, min(255, round(g))),
        max(0, min(255, round(b)))
    )

def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Конвертирует HSV в RGB."""
    h = h % 360
    s = max(0, min(100, s)) / 100
    v = max(0, min(100, v)) / 100
    
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return normalize_rgb(
        (r + m) * 255,
        (g + m) * 255,
        (b + m) * 255
    )

def generate_gradient(start_color: tuple[int, int, int], end_color: tuple[int, int, int], steps: int) -> list[tuple[int, int, int]]:
    """Генерирует градиент между двумя цветами."""
    r1, g1, b1 = start_color
    r2, g2, b2 = end_color
    
    gradient = []
    for i in range(steps):
        t = i / (steps - 1)
        r = r1 + (r2 - r1) * t
        g = g1 + (g2 - g1) * t
        b = b1 + (b2 - b1) * t
        gradient.append(normalize_rgb(r, g, b))
    
    return gradient

def generate_color_schemes(color: tuple[int, int, int]) -> dict[str, list[tuple[int, int, int]]]:
    """Генерирует различные цветовые схемы на основе входного цвета."""
    h, s, v = rgb_to_hsv(*color)
    
    # Получаем дополнительный цвет
    complementary = hsv_to_rgb((h + 180) % 360, s, v)
    
    # Получаем темную и светлую версии цвета
    dark = hsv_to_rgb(h, min(s * 1.2, 100), max(v * 0.6, 0))
    light = hsv_to_rgb(h, max(s * 0.8, 0), min(v * 1.4, 100))
    
    # Названия схем берем из переводов
    schemes = {
        tr("monochromatic"): [
            hsv_to_rgb(h, max(s * 0.5, 0), v),
            hsv_to_rgb(h, max(s * 0.75, 0), v),
            color,
            hsv_to_rgb(h, min(s * 1.25, 100), v),
            hsv_to_rgb(h, min(s * 1.5, 100), v)
        ],
        tr("analogous"): [
            hsv_to_rgb((h - 30) % 360, s, v),
            hsv_to_rgb((h - 15) % 360, s, v),
            color,
            hsv_to_rgb((h + 15) % 360, s, v),
            hsv_to_rgb((h + 30) % 360, s, v)
        ],
        tr("complementary"): [
            hsv_to_rgb(h, max(s * 0.8, 0), min(v * 1.1, 100)),
            color,
            complementary,
            hsv_to_rgb((h + 180) % 360, max(s * 0.8, 0), min(v * 1.1, 100))
        ],
        tr("triadic"): [
            color,
            hsv_to_rgb((h + 120) % 360, s, v),
            hsv_to_rgb((h + 240) % 360, s, v)
        ],
        tr("light_gradient"): generate_gradient(color, light, 5),
        tr("dark_gradient"): generate_gradient(color, dark, 5),
        tr("complementary_gradient"): generate_gradient(color, complementary, 5),
        tr("double_gradient"): generate_gradient(dark, light, 5),
        tr("rainbow_gradient"): [
            hsv_to_rgb((h + angle) % 360, s, v)
            for angle in [0, 60, 120, 180, 240, 300]
        ]
    }
    return schemes

class ColorButton(QFrame):
    clicked = pyqtSignal()
    
    def __init__(self, title_key):
        super().__init__()
        self.color = QColor(255, 255, 255)
        self.title_key = title_key
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMaximumSize(300, 300)
        self.setObjectName("colorButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.updateStyle()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем заголовок
        self.title_label = QLabel(tr(title_key))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                background-color: rgba(0, 0, 0, 0.6);
                border-radius: 8px;
                margin: 8px;
            }
        """)
        
        layout.addWidget(self.title_label)
        
        # Устанавливаем начальную подсказку
        self.updateTooltip()
    
    def updateStyle(self):
        self.setStyleSheet(f"""
            QFrame#colorButton {{
                background-color: {self.color.name()};
                border-radius: 15px;
                border: 2px solid #3D3D3D;
            }}
            QFrame#colorButton:hover {{
                border: 2px solid #5D5D5D;
            }}
        """)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
    
    def setColor(self, color: QColor):
        if color.isValid():
            self.color = color
            self.updateStyle()
            self.updateTooltip()
    
    def get_rgb(self):
        return (self.color.red(), self.color.green(), self.color.blue())
    
    def updateTooltip(self):
        """Обновляет всплывающую подсказку с информацией о цвете"""
        rgb = self.get_rgb()
        r, g, b = rgb
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        h, s, v = rgb_to_hsv(r, g, b)
        L, a, b_value = rgb_to_lab(r, g, b)
        
        tooltip = f"""
            <b>{tr("color_info")}</b><br>
            <table>
                <tr><td>RGB:</td><td>{r}, {g}, {b}</td></tr>
                <tr><td>HEX:</td><td>{hex_color}</td></tr>
                <tr><td>HSV:</td><td>{int(h)}°, {int(s)}%, {int(v)}%</td></tr>
                <tr><td>LAB:</td><td>{int(L)}, {int(a)}, {int(b_value)}</td></tr>
            </table>
        """
        
        self.setToolTip(tooltip)
        self.setToolTipDuration(10000)  # Показывать подсказку 10 секунд
        
    def updateTranslation(self):
        """Обновляет текст кнопки при смене языка"""
        self.title_label.setText(tr(self.title_key))
        self.updateTooltip()

class ColorAnalysisTab(QWidget):
    def __init__(self, title_key: str, description_key: str, result_callback):
        super().__init__()
        self.title_key = title_key
        self.description_key = description_key
        self.result_callback = result_callback
        
        # Создаем и настраиваем диалог выбора цвета
        self.color_dialog = QColorDialog(self)
        self.color_dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        self.color_dialog.setStyleSheet("""
            QColorDialog {
                background-color: #1E1E1E;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            QLabel {
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QPushButton:pressed {
                background-color: #2D2D2D;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3D3D3D;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3D3D3D;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(30)
        
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(30)
        
        # Заголовок
        self.title_label = QLabel(tr(title_key))
        self.title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #FFFFFF;
            margin-bottom: 20px;
        """)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Описание
        self.description_label = QLabel(tr(description_key))
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setStyleSheet("""
            font-size: 16px;
            color: #B0B0B0;
            margin: 20px;
            line-height: 150%;
        """)
        self.description_label.setWordWrap(True)
        
        # Результат
        self.result_label = QLabel(tr("select_colors"))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #FFFFFF;
            padding: 25px;
            background-color: rgba(61, 61, 61, 0.6);
            border-radius: 15px;
            margin: 20px;
        """)
        
        content_layout.addWidget(self.title_label)
        content_layout.addWidget(self.description_label)
        
        # Кнопки выбора цвета в контейнере
        colors_container = QWidget()
        colors_layout = QHBoxLayout(colors_container)
        colors_layout.setSpacing(40)
        colors_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.color_button1 = ColorButton("color_1")
        self.color_button2 = ColorButton("color_2")
        
        self.color_button1.clicked.connect(lambda: self.pick_color(self.color_button1))
        self.color_button2.clicked.connect(lambda: self.pick_color(self.color_button2))
        
        colors_layout.addWidget(self.color_button1)
        colors_layout.addWidget(self.color_button2)
        
        content_layout.addWidget(colors_container)
        content_layout.addWidget(self.result_label)
        content_layout.addStretch()
        
        layout.addWidget(content_widget)
    
    def pick_color(self, button):
        self.color_dialog.setCurrentColor(button.color)
        if self.color_dialog.exec() == QColorDialog.DialogCode.Accepted:
            color = self.color_dialog.currentColor()
            button.setColor(color)
            self.update_result()
    
    def update_result(self):
        self.result_label.setText(
            self.result_callback(
                self.color_button1.get_rgb(),
                self.color_button2.get_rgb()
            )
        )
        
    def updateTranslation(self):
        """Обновляет все текстовые элементы при смене языка"""
        self.title_label.setText(tr(self.title_key))
        self.description_label.setText(tr(self.description_key))
        if self.result_label.text() == tr("select_colors", lang_manager.current_language):
            self.result_label.setText(tr("select_colors"))
        else:
            # Если у нас уже есть результат, обновляем его
            self.update_result()
        
        # Обновляем дочерние виджеты
        self.color_button1.updateTranslation()
        self.color_button2.updateTranslation()

class ColorSchemeDisplay(QWidget):
    def __init__(self, scheme_name: str, colors: list[tuple[int, int, int]], parent=None):
        super().__init__(parent)
        self.scheme_name = scheme_name
        self.colors = colors
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Заголовок и кнопки
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Название схемы
        name_label = QLabel(scheme_name)
        name_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: white;
            padding: 5px;
        """)
        header_layout.addWidget(name_label)
        
        # Добавляем кнопки экспорта
        export_css_btn = QPushButton("CSS")
        export_css_btn.setToolTip(tr("export_to_css"))
        export_css_btn.setFixedSize(58, 35)
        export_css_btn.clicked.connect(self.export_to_css)
        
        export_scss_btn = QPushButton("SCSS")
        export_scss_btn.setToolTip(tr("export_to_scss"))
        export_scss_btn.setFixedSize(58, 35)
        export_scss_btn.clicked.connect(self.export_to_scss)
        
        header_layout.addStretch()
        header_layout.addWidget(export_css_btn)
        header_layout.addWidget(export_scss_btn)
        
        layout.addWidget(header_container)
        
        # Цветовые квадраты в контейнере с градиентом
        colors_container = QWidget()
        colors_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        colors_layout = QHBoxLayout(colors_container)
        colors_layout.setSpacing(0)  # Убираем отступы для плавного градиента
        
        for i, color in enumerate(colors):
            color_square = QFrame()
            color_square.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            color_square.setMinimumSize(50, 50)
            
            # Добавляем скругление только для крайних элементов
            border_radius = ""
            if i == 0:
                border_radius = "border-top-left-radius: 5px; border-bottom-left-radius: 5px;"
            elif i == len(colors) - 1:
                border_radius = "border-top-right-radius: 5px; border-bottom-right-radius: 5px;"
            
            color_square.setStyleSheet(f"""
                background-color: rgb{color};
                {border_radius}
                border: none;
            """)
            
            # Добавляем всплывающую подсказку
            r, g, b = color
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            h, s, v = rgb_to_hsv(r, g, b)
            L, a, b_value = rgb_to_lab(r, g, b)
            
            tooltip = f"""
                <b>{tr("color_number", i+1)}</b><br>
                <table>
                    <tr><td>RGB:</td><td>{r}, {g}, {b}</td></tr>
                    <tr><td>HEX:</td><td>{hex_color}</td></tr>
                    <tr><td>HSV:</td><td>{int(h)}°, {int(s)}%, {int(v)}%</td></tr>
                    <tr><td>LAB:</td><td>{int(L)}, {int(a)}, {int(b_value)}</td></tr>
                </table>
            """
            
            color_square.setToolTip(tooltip)
            color_square.setToolTipDuration(10000)  # Показывать подсказку 10 секунд
            
            # Устанавливаем курсор для указания интерактивности
            color_square.setCursor(Qt.CursorShape.PointingHandCursor)
            
            # Добавляем контекстное меню для копирования
            color_square.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            color_square.customContextMenuRequested.connect(
                lambda pos, col=color: self.show_color_menu(pos, col)
            )
            
            colors_layout.addWidget(color_square)
        
        layout.addWidget(colors_container)
        
        # RGB значения
        rgb_container = QWidget()
        rgb_layout = QHBoxLayout(rgb_container)
        rgb_layout.setSpacing(10)
        
        for color in colors:
            rgb_label = QLabel(f"RGB: {color}")
            rgb_label.setStyleSheet("""
                color: #BBB;
                font-size: 10px;
            """)
            rgb_layout.addWidget(rgb_label, 1)
        
        layout.addWidget(rgb_container)
    
    def show_color_menu(self, position, color):
        """Показывает контекстное меню для цвета"""
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3D3D3D;
                border-radius: 5px;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #3D3D3D;
            }
        """)
        
        copy_rgb = menu.addAction(tr("copy_rgb", r, g, b))
        copy_hex = menu.addAction(tr("copy_hex", hex_color))
        
        action = menu.exec(self.mapToGlobal(position))
        
        if action == copy_rgb:
            QApplication.clipboard().setText(f"{r}, {g}, {b}")
        elif action == copy_hex:
            QApplication.clipboard().setText(hex_color)
    
    def export_to_css(self):
        """Экспортирует цветовую схему в CSS файл"""
        # Создаем CSS контент
        css_content = f"/* {self.scheme_name} - Color Scheme */\n:root {{\n"
        
        for i, (r, g, b) in enumerate(self.colors):
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            css_content += f"  --color-{i+1}: {hex_color};\n"
        
        css_content += "}\n"
        
        # Предлагаем пользователю сохранить файл
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            tr("save_css"), 
            f"{self.scheme_name.lower().replace(' ', '_')}.css", 
            "CSS Files (*.css)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(css_content)
                self.show_message(tr("export_success"), tr("css_saved", file_path))
            except Exception as e:
                self.show_message(tr("export_error"), tr("file_save_error", str(e)), error=True)
    
    def export_to_scss(self):
        """Экспортирует цветовую схему в SCSS файл"""
        # Создаем SCSS контент с переменными
        scss_content = f"// {self.scheme_name} - Color Scheme\n"
        
        # Добавляем переменные SCSS
        for i, (r, g, b) in enumerate(self.colors):
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            scss_content += f"$color-{i+1}: {hex_color};\n"
        
        # Добавляем карту цветов
        scss_content += "\n// Карта цветов\n$colors: (\n"
        for i, (r, g, b) in enumerate(self.colors):
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            comma = "," if i < len(self.colors) - 1 else ""
            scss_content += f"  'color-{i+1}': {hex_color}{comma}\n"
        scss_content += ");\n\n"
        
        # Добавляем функцию получения цвета
        scss_content += "// Функция для получения цвета из карты\n"
        scss_content += "@function get-color($name) {\n"
        scss_content += "  @return map-get($colors, $name);\n"
        scss_content += "}\n"
        
        # Предлагаем пользователю сохранить файл
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            tr("save_scss"), 
            f"{self.scheme_name.lower().replace(' ', '_')}.scss", 
            "SCSS Files (*.scss)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(scss_content)
                self.show_message(tr("export_success"), tr("scss_saved", file_path))
            except Exception as e:
                self.show_message(tr("export_error"), tr("file_save_error", str(e)), error=True)
    
    def show_message(self, title, message, error=False):
        """Показывает информационное сообщение"""
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Information)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2D2D2D;
            }
            QLabel {
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
        """)
        msg_box.exec()

class ColorSchemesTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Создаем и настраиваем диалог выбора цвета
        self.color_dialog = QColorDialog(self)
        self.color_dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        self.color_dialog.setStyleSheet("""
            QColorDialog {
                background-color: #1E1E1E;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            QLabel {
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QPushButton:pressed {
                background-color: #2D2D2D;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3D3D3D;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3D3D3D;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Заголовок
        self.title = QLabel(tr("schemes_title"))
        self.title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: white;
            margin-bottom: 20px;
        """)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Описание
        self.description = QLabel(tr("schemes_desc"))
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setStyleSheet("""
            font-size: 16px;
            color: #BBBBBB;
            margin: 20px;
        """)
        self.description.setWordWrap(True)
        
        # Кнопка выбора цвета перед добавлением action_container
        self.color_button = ColorButton("base_color")
        self.color_button.clicked.connect(self.pick_color)
        
        # Теперь создаем контейнер действий
        action_container = QWidget()
        action_layout = QHBoxLayout(action_container)
        action_layout.setContentsMargins(0, 10, 0, 10)
        
        self.extract_from_image_btn = QPushButton(tr("extract_from_image"))
        self.extract_from_image_btn.clicked.connect(self.extract_colors_from_image_dialog)
        action_layout.addWidget(self.extract_from_image_btn)
        
        # Добавляем все компоненты в правильном порядке
        main_layout.addWidget(self.title)
        main_layout.addWidget(self.description)
        main_layout.addWidget(self.color_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(action_container)  # Добавляем после кнопки цвета
        
        # Область для схем с прокруткой
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2C2C2C;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.schemes_widget = QWidget()
        self.schemes_layout = QVBoxLayout(self.schemes_widget)
        self.schemes_layout.setSpacing(20)
        scroll_area.setWidget(self.schemes_widget)
        
        # Добавляем всё в главный layout
        main_layout.addWidget(scroll_area)
        
        # Генерируем начальные схемы с цветом по умолчанию
        self.color_button.setColor(QColor(100, 150, 200))
        self.update_schemes()
    
    def pick_color(self):
        self.color_dialog.setCurrentColor(self.color_button.color)
        if self.color_dialog.exec() == QColorDialog.DialogCode.Accepted:
            color = self.color_dialog.currentColor()
            self.color_button.setColor(color)
            self.update_schemes()
    
    def update_schemes(self):
        # Очищаем предыдущие схемы
        for i in reversed(range(self.schemes_layout.count())): 
            self.schemes_layout.itemAt(i).widget().setParent(None)
        
        # Генерируем новые схемы
        schemes = generate_color_schemes(self.color_button.get_rgb())
        for name, colors in schemes.items():
            self.schemes_layout.addWidget(ColorSchemeDisplay(name, colors))

    def copy_color_to_clipboard(self, color_tuple):
        """Копирование цвета в разных форматах в буфер обмена"""
        r, g, b = color_tuple
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        QApplication.clipboard().setText(hex_color)

    def extract_colors_from_image(self, image_path, num_colors=5):
        """Извлекает доминирующие цвета из изображения"""
        # Проверяем наличие всех необходимых библиотек
        if not has_pil or not has_numpy or not has_sklearn:
            # Если хотя бы одна библиотека отсутствует, выдаем сообщение
            QMessageBox.critical(
                self, 
                tr("missing_dependencies"), 
                tr("dependencies_msg")
            )
            return []
            
        # Устанавливаем переменную среды для ограничения числа процессоров
        os.environ["LOKY_MAX_CPU_COUNT"] = "1"
        
        # Загрузка и преобразование изображения
        img = Image.open(image_path)
        
        # Принудительно конвертируем в RGB режим
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((100, 100))  # уменьшаем для ускорения
        img_array = np.array(img)
        
        # Проверка формы массива
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Извлекаем только RGB каналы (игнорируем альфа-канал, если есть)
            pixels = img_array[:, :, 0:3].reshape(-1, 3)
        else:
            # Сообщение об ошибке, если формат изображения неподдерживаемый
            raise ValueError(f"Неподдерживаемый формат изображения. Форма массива: {img_array.shape}")
        
        # Кластеризация для поиска доминирующих цветов
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        
        # Возвращаем центры кластеров как цвета
        colors = [(int(r), int(g), int(b)) for r, g, b in kmeans.cluster_centers_]
        return colors

    def save_color_scheme(self, scheme_name, colors):
        """Сохраняет цветовую схему в файл JSON"""
        import json
        from datetime import datetime
        
        # Преобразуем цвета в формат для сохранения
        colors_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        
        # Создаем запись
        scheme = {
            "name": scheme_name,
            "colors": colors_hex,
            "date_created": datetime.now().isoformat()
        }
        
        # Загружаем существующие схемы
        try:
            with open("color_schemes.json", "r") as f:
                saved_schemes = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            saved_schemes = []
        
        # Добавляем новую схему
        saved_schemes.append(scheme)
        
        # Сохраняем обратно
        with open("color_schemes.json", "w") as f:
            json.dump(saved_schemes, f, indent=2)

    def create_color_tooltip(self, rgb_color):
        """Создает информативную подсказку для цвета"""
        r, g, b = rgb_color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        h, s, v = rgb_to_hsv(r, g, b)
        L, a, b_value = rgb_to_lab(r, g, b)
        
        return f"RGB: {r}, {g}, {b}\nHEX: {hex_color}\nHSV: {int(h)}°, {int(s)}%, {int(v)}%\nLAB: {int(L)}, {int(a)}, {int(b_value)}"

    def export_to_css(self, scheme_name, colors):
        """Экспортирует цветовую схему в CSS переменные"""
        css_content = f"/* {scheme_name} - Color Scheme */\n:root {{\n"
        
        for i, (r, g, b) in enumerate(colors):
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            css_content += f"  --color-{i+1}: {hex_color};\n"
        
        css_content += "}\n"
        return css_content

    def extract_colors_from_image_dialog(self):
        """Открывает диалог выбора изображения и извлекает из него цвета"""
        # Проверяем наличие необходимых библиотек
        if not has_pil or not has_numpy or not has_sklearn:
            QMessageBox.critical(
                self, 
                tr("missing_dependencies"), 
                tr("dependencies_msg")
            )
            return
        
        # Открываем диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            tr("select_image"), 
            "", 
            "Изображения (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not file_path:
            return
        
        # Запрашиваем количество цветов для извлечения
        num_colors, ok = QInputDialog.getInt(
            self, 
            tr("colors_count"), 
            tr("colors_count_msg"), 
            5, 2, 10, 1
        )
        
        if not ok:
            return
        
        # Показываем индикатор загрузки
        self.setCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Извлекаем цвета из изображения
            colors = self.extract_colors_from_image(file_path, num_colors)
            
            if not colors:  # Если extract_colors_from_image вернул пустой список из-за отсутствия библиотек
                self.setCursor(Qt.CursorShape.ArrowCursor)
                return
                
            # Создаем схему из извлеченных цветов
            scheme_name = tr("colors_from", os.path.basename(file_path))
            
            # Очищаем предыдущие схемы и показываем только эту
            for i in reversed(range(self.schemes_layout.count())): 
                self.schemes_layout.itemAt(i).widget().setParent(None)
            
            # Добавляем извлеченную схему
            self.schemes_layout.addWidget(ColorSchemeDisplay(scheme_name, colors, self))
            
            # Устанавливаем первый цвет как базовый
            if colors:
                r, g, b = colors[0]
                self.color_button.setColor(QColor(r, g, b))
        
        except Exception as e:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.show_message(tr("image_error"), str(e), error=True)
        
        # Восстанавливаем курсор
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def show_message(self, title, message, error=False):
        """Показывает информационное сообщение"""
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Information)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2D2D2D;
            }
            QLabel {
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
        """)
        msg_box.exec()
        
    def updateTranslation(self):
        """Обновляет текст при смене языка"""
        self.title.setText(tr("schemes_title"))
        self.description.setText(tr("schemes_desc"))
        self.extract_from_image_btn.setText(tr("extract_from_image"))
        self.color_button.updateTranslation()
        
        # Обновляем названия схем
        self.update_schemes()

class ColorHarmonyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("app_title"))
        self.setMinimumSize(800, 600)
        self.resize(1024, 768)
        
        # Список доступных языков для циклического переключения
        self.available_languages = ["ru", "en", "zh"]
        
        # Основные цвета темы
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QWidget {
                background-color: #1E1E1E;
            }
            QLabel {
                color: #E0E0E0;
            }
            QTabWidget::pane {
                border: none;
                background-color: #1E1E1E;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #E0E0E0;
                padding: 12px 30px;
                margin: 4px;
                border-radius: 8px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background-color: #3D3D3D;
                color: #FFFFFF;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #353535;
            }
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QPushButton:pressed {
                background-color: #2D2D2D;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2D2D2D;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #4D4D4D;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5D5D5D;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: #2D2D2D;
                border-radius: 6px;
            }
            QToolTip {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #3D3D3D;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        
        # Создаем главный виджет для размещения всех элементов
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Верхняя панель с кнопкой переключения языка
        top_panel = QWidget()
        top_panel.setStyleSheet("background-color: #1E1E1E;")
        top_panel.setFixedHeight(40)
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(15, 5, 15, 5)
        
        # Создаем кнопку переключения языка
        self.lang_button = QPushButton(self.get_language_display_name())
        self.lang_button.setStyleSheet("""
            QPushButton {
                background-color: #3D3D3D;
                color: #E0E0E0;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QPushButton:pressed {
                background-color: #2D2D2D;
            }
        """)
        self.lang_button.clicked.connect(self.cycle_language)
        
        top_layout.addWidget(self.lang_button)
        top_layout.addStretch()
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.tabs.setContentsMargins(0, 0, 0, 0)
        
        # Вкладка сходства цветов
        self.similarity_tab = ColorAnalysisTab(
            "similarity_title",
            "similarity_desc",
            lambda c1, c2: tr("similarity_result", calculate_color_similarity(c1, c2))
        )
        
        # Вкладка гармонии цветов
        self.harmony_tab = ColorAnalysisTab(
            "harmony_title",
            "harmony_desc",
            lambda c1, c2: self.format_harmony_result(*calculate_harmony_score(c1, c2))
        )
        
        # Вкладка цветовых схем
        self.schemes_tab = ColorSchemesTab()
        
        self.tabs.addTab(self.similarity_tab, tr("similarity_tab"))
        self.tabs.addTab(self.harmony_tab, tr("harmony_tab"))
        self.tabs.addTab(self.schemes_tab, tr("schemes_tab"))
        
        # Добавляем всё в главный layout
        main_layout.addWidget(top_panel)
        main_layout.addWidget(self.tabs)
        
        self.setCentralWidget(main_widget)
    
    def get_language_display_name(self):
        """Возвращает отображаемое имя текущего языка для кнопки"""
        language_names = {
            "ru": "Русский",
            "en": "English",
            "zh": "中文"
        }
        return language_names.get(lang_manager.current_language, "Language")
    
    def cycle_language(self):
        """Циклически меняет язык по нажатию кнопки"""
        # Получаем индекс текущего языка
        current_index = self.available_languages.index(lang_manager.current_language) if lang_manager.current_language in self.available_languages else 0
        
        # Вычисляем индекс следующего языка (циклически)
        next_index = (current_index + 1) % len(self.available_languages)
        
        # Получаем код следующего языка
        next_language = self.available_languages[next_index]
        
        # Устанавливаем новый язык
        if lang_manager.set_language(next_language):
            # Обновляем текст кнопки
            self.lang_button.setText(self.get_language_display_name())
            # Обновляем интерфейс
            self.updateTranslation()
    
    def format_harmony_result(self, score: float, harmony_type: str) -> str:
        rating_key = "excellent" if score >= 80 else "good" if score >= 60 else "satisfactory" if score >= 40 else "poor"
        rating = tr(rating_key)
        return tr("harmony_type", harmony_type) + "\n" + tr("harmony_score", score, rating)
    
    def updateTranslation(self):
        """Обновляет все переводимые элементы интерфейса"""
        # Обновляем заголовок окна
        self.setWindowTitle(tr("app_title"))
        
        # Обновляем названия вкладок
        self.tabs.setTabText(0, tr("similarity_tab"))
        self.tabs.setTabText(1, tr("harmony_tab"))
        self.tabs.setTabText(2, tr("schemes_tab"))
        
        # Обновляем содержимое вкладок
        self.similarity_tab.updateTranslation()
        self.harmony_tab.updateTranslation()
        self.schemes_tab.updateTranslation()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorHarmonyApp()
    window.show()
    sys.exit(app.exec()) 
