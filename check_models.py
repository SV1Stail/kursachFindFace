import cv2
import os

# Пути к папке с моделями
MODEL_DIR = 'models'

# Файлы моделей
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
AGE_PROTO = os.path.join(MODEL_DIR, 'age_deploy.prototxt')
AGE_MODEL = os.path.join(MODEL_DIR, 'age_net.caffemodel')
GENDER_PROTO = os.path.join(MODEL_DIR, 'gender_deploy.prototxt')
GENDER_MODEL = os.path.join(MODEL_DIR, 'gender_net.caffemodel')
EMOTION_MODEL = os.path.join(MODEL_DIR, 'emotion-ferplus-8.onnx')

# Проверка загрузки каскада Хаара
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError('Ошибка загрузки каскада Хаара')
    print('Каскад Хаара успешно загружен')
except Exception as e:
    print(f'Ошибка каскада Хаара: {e}')

# Проверка загрузки модели возраста
try:
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    print('Модель возраста успешно загружена')
except Exception as e:
    print(f'Ошибка загрузки модели возраста: {e}')

# Проверка загрузки модели пола
try:
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    print('Модель пола успешно загружена')
except Exception as e:
    print(f'Ошибка загрузки модели пола: {e}')

# Проверка загрузки модели эмоций
try:
    emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)
    print('Модель эмоций успешно загружена')
except Exception as e:
    print(f'Ошибка загрузки модели эмоций: {e}')
