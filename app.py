from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Пути к моделям
MODEL_DIR = 'models'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
AGE_PROTO = os.path.join(MODEL_DIR, 'age_deploy.prototxt')
AGE_MODEL = os.path.join(MODEL_DIR, 'age_net.caffemodel')
GENDER_PROTO = os.path.join(MODEL_DIR, 'gender_deploy.prototxt')
GENDER_MODEL = os.path.join(MODEL_DIR, 'gender_net.caffemodel')
EMOTION_MODEL = os.path.join(MODEL_DIR, 'emotion-ferplus-8.onnx')

# Категории
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# Загрузка моделей
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)

def analyze_face(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (64, 64))

    blob_emotion = cv2.dnn.blobFromImage(face_gray, 1.0 / 255, (64, 64),
                                         (0, 0, 0), swapRB=False, crop=False)
    emotion_net.setInput(blob_emotion)
    emotion_preds = emotion_net.forward()
    emotion = EMOTIONS[np.argmax(emotion_preds)]

    return gender, age, emotion

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file uploaded.', 400
        
        file = request.files['image']
        if file.filename == '':
            return 'No selected file.', 400
        
        # Считываем изображение
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            gender, age, emotion = analyze_face(face_img)

            label = f"{gender}, {age}, {emotion}"
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Сохраняем обработанное изображение
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)

        return render_template('index.html', filename=filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port)
