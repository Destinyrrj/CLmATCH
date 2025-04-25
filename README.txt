Для запуска на Windows нужно запустить exe файл. (НЕ РЕКОМЕНДУЕТСЯ)

Для запуска на Linux нужно запустить исполняемый файл из папки Linux. (Скоро будет)

Для ручной сборки на Linux нужно :
1. chmod +x pyinstaller
2. python(3) pyinstaller build_app.spec
3. ./dist/ColorHarmonyAnalyzer/ColorHarmonyAnalyzer или нажать на иконку в меню приложений.

Для ручной сборки на Windows нужно : 
1. pip install --user pyinstaller PyQt6==6.3.1
2. pyinstaller build_app.spec
3. ./dist/ColorHarmonyAnalyzer/ColorHarmonyAnalyzer.exe или нажать на иконку в меню приложений.

Для запуска из исходников нужно : (РЕКОМЕНДУЕТСЯ)
1. python(3) -m pip install --user PyQt6==6.3.1
2. python color_harmony_app.py
