# Проект реализован в рамках программы Intel CV Summer Camp 2019.

# Smart Glove

## Описание проекта

Умная перчатка – приспособление для распознавания различных бытовых объектов для помощи людям с ограниченными способностями. Перчатка должна выполнять функции локализации и распознавания предметов быта с помощью переносимой системы видеонаблюдения.

## Постановка задачи

1. Реализовать систему детектирования объектов на кадре.
2. Отслеживать найденные предметы.
3. Навигация по конкретному объекту или отображение классов объектов.

## Требования к проекту

1. OpenCV 4.1.0 с модулями xfeatures2d из opencv_contrib, модуль dnn;
2. Mobilenet-ssd v2 coco
2. Пакет SFML Audio v2.4.2.
3. Пакет Microsoft.CognitiveServices.Speech v.1.6.0.
4. Две usb-камеры, либо видеофайл, полученный с помощью съемки через стереоустановку
5. Файл с описанием параметров камер и их положением относительно друг друга (Калибровочный файл). Калибровка соотетствующих камер должна быть получена с помощью приложения калибровки (будет загружено позже).
  
## Описание модулей

1. Модуль Calibration позволяет откалибровать одну либо две камеры. На вход требуется видео или массив изображений. В модуле реализовано сохранение полученных параметров в файл, а также его последующее считывание с возможностью вычисления дополнительных параметров, на основе уже полученных ранее.
2. Модуль DnnDetector позволяет считывать параметры для mobilenet-ssd v1 и v2(coco) и в последующем детектировать объекты на кадре, записывая их в массив.
3. Модуль TrackingByMatching реализован на основе сопоставления данных, полученных с помощью детектора. Возвращает массив трекируемых объектов.
4. Модуль MatchFeatures позволяет находить особые точки на query image и training image и сопоставлять их между собой.

## Навигация

1. [Подключаемые модули находятся в папке include](include/)
2. [Исходные файлы модулей в папке src](src/)
3. [Основная функция в папке SmartGlove](SmartGlove/)

## Текущее состояние

Приложение позволяет определять объекты на левом кадре, трекировать их, а также определять расстояние, вычисляя глубину с помощью точек, полученных с помощью ORB. Также возможен выбор отслеживаемых классов объектов, либо выбор конкретного объекта и навигация до него.

## Текущие проблемы

1. Детектор работает медленно и не всегда верно определяет предметы.
2. Трекер зачастую дает сбои. Сильно привязан к детектору. Требуется подбор коэффициентов.
3. Сопоставление объектов на левом и правом кадрах происходит не всегда и работает недостаточно быстро. Возможно, проблема связана с качеством камер и калибровки.
4. Области, полученные с помощью сопоставления объектов на кадрах, не всегда верны.
5. Голосовое управление и голосовая навигация не работают из-за неправильной установки Microsoft.CognitiviveServices.Speech. Возникает проблема при загрузке. Причина не выяснена.
