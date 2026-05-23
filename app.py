import os
import io
from datetime import datetime

import numpy as np
import onnxruntime as ort
from PIL import Image

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ── APP ──────────────────────────────────────────────────────────────
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ictino.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_ictino')

db = SQLAlchemy(app)

# ── MODELO ONNX (carga única al arrancar) ────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'best.onnx')

session     = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
INPUT_NAME  = session.get_inputs()[0].name
INPUT_SHAPE = session.get_inputs()[0].shape   # [1, 3, H, W]

# Si el shape tiene dimensiones dinámicas ('height'/'width'), usamos 640
IMG_H = INPUT_SHAPE[2] if isinstance(INPUT_SHAPE[2], int) else 640
IMG_W = INPUT_SHAPE[3] if isinstance(INPUT_SHAPE[3], int) else 640

# ── MODELOS DE BASE DE DATOS ─────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'usuarios'
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80),  unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class DetectionRecord(db.Model):
    __tablename__ = 'historial_detecciones'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=True)
    fecha         = db.Column(db.String(50), nullable=False)
    latitud       = db.Column(db.Float, nullable=True)
    longitud      = db.Column(db.Float, nullable=True)
    max_confianza = db.Column(db.Float, nullable=False)

# ── PREPROCESAMIENTO ─────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    Redimensiona la imagen al tamaño esperado por el modelo,
    normaliza a [0, 1] y añade la dimensión de batch.
    Devuelve un array (1, 3, H, W) float32.
    """
    img = pil_img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    arr = arr.transpose(2, 0, 1)                     # (3, H, W)
    return arr[np.newaxis]                            # (1, 3, H, W)

# ── POSTPROCESAMIENTO ────────────────────────────────────────────────
def postprocess(outputs, orig_w: int, orig_h: int, conf_thresh: float = 0.25):
    """
    Interpreta la salida del modelo ONNX exportado con ultralytics.
    Soporta el formato YOLOv8: (1, 4+nc, num_anchors).
    Las coordenadas devueltas son píxeles absolutos de la imagen original.
    """
    raw   = outputs[0]   # (1, 4+nc, num_anchors)  o  (1, num_anchors, 4+nc)
    preds = raw[0]       # quita dimensión de batch → (4+nc, num_anchors)

    # Ultralytics exporta (4+nc, anchors); si viene al revés lo transponemos
    if preds.shape[0] > preds.shape[1]:
        preds = preds.T  # ahora (num_anchors, 4+nc)

    detections = []
    for row in preds:
        # row: [cx, cy, w, h, conf_cls0, conf_cls1, ...]
        class_scores = row[4:]
        conf = float(class_scores.max())
        if conf < conf_thresh:
            continue

        cx, cy, w, h = row[:4]

        # Coordenadas en escala del tamaño de entrada del modelo → píxeles originales
        x1 = (cx - w / 2) / IMG_W * orig_w
        y1 = (cy - h / 2) / IMG_H * orig_h
        x2 = (cx + w / 2) / IMG_W * orig_w
        y2 = (cy + h / 2) / IMG_H * orig_h

        # Clampear dentro de los bordes de la imagen
        x1 = max(0.0, min(float(x1), float(orig_w)))
        y1 = max(0.0, min(float(y1), float(orig_h)))
        x2 = max(0.0, min(float(x2), float(orig_w)))
        y2 = max(0.0, min(float(y2), float(orig_h)))

        detections.append({
            "box":        [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "class":      "pothole"
        })

    # Ordenar de mayor a menor confianza
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections

# ── RUTAS ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibieron datos"}), 400

    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({"error": "Por favor ingresa usuario y contraseña."}), 400

    user = db.session.execute(
        db.select(User).filter_by(username=username)
    ).scalar_one_or_none()

    if user and check_password_hash(user.password, password):
        return jsonify({"user_id": user.id, "status": "success"}), 200

    return jsonify({"error": "Usuario o contraseña incorrectos."}), 401


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibieron datos"}), 400

    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({"error": "Por favor ingresa usuario y contraseña."}), 400

    existing = db.session.execute(
        db.select(User).filter_by(username=username)
    ).scalar_one_or_none()
    if existing:
        return jsonify({"error": "El nombre de usuario ya está registrado."}), 400

    new_user = User(
        username=username,
        password=generate_password_hash(password, method='scrypt')
    )
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status": "success", "message": "Usuario creado correctamente"}), 201
    except Exception:
        db.session.rollback()
        return jsonify({"error": "Error al guardar el usuario en la base de datos."}), 500


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "Archivo vacío"}), 400

    try:
        img            = Image.open(io.BytesIO(file.read())).convert('RGB')
        orig_w, orig_h = img.size

        inp        = preprocess(img)
        outputs    = session.run(None, {INPUT_NAME: inp})
        detections = postprocess(outputs, orig_w, orig_h)

        # Guardar resumen en BD si hay sesión activa
        user_id = request.form.get('user_id')
        if user_id and detections:
            max_conf = max(d['confidence'] for d in detections)
            db.session.add(DetectionRecord(
                user_id=int(user_id),
                fecha=datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
                latitud=None,
                longitud=None,
                max_confianza=max_conf
            ))
            db.session.commit()

        return jsonify({"detections": detections})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


# ── INICIALIZACIÓN ───────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
