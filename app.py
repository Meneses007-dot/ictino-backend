import os
import io
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from ultralytics import YOLO

# ── APP ─────────────────────────────────────────────────────────────
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ictino.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_ictino')

db = SQLAlchemy(app)

# ── MODELO  (carga única al arrancar) ────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'best.pt')
model = YOLO(MODEL_PATH)

# ── MODELOS DE BASE DE DATOS ─────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'usuarios'
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80),  unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class DetectionRecord(db.Model):
    __tablename__ = 'historial_detecciones'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=True)
    fecha        = db.Column(db.String(50), nullable=False)
    latitud      = db.Column(db.Float, nullable=True)
    longitud     = db.Column(db.Float, nullable=True)
    max_confianza = db.Column(db.Float, nullable=False)

# ── RUTAS ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Sirve el frontend principal."""
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
    """
    Recibe multipart/form-data con:
      - image   : archivo de imagen
      - user_id : (opcional) id del usuario autenticado

    Devuelve JSON:
      {
        "detections": [
          { "box": [x1, y1, x2, y2], "confidence": 0.87, "class": "pothole" },
          ...
        ]
      }
    Las coordenadas de box son píxeles absolutos de la imagen original.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "Archivo vacío"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        results = model(img, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "box":        [x1, y1, x2, y2],
                "confidence": conf,
                "class":      "pothole"
            })

        # Guardar resumen en base de datos si hay sesión activa
        user_id = request.form.get('user_id')
        if user_id and detections:
            max_conf = max(d['confidence'] for d in detections)
            record = DetectionRecord(
                user_id=int(user_id),
                fecha=datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
                latitud=None,
                longitud=None,
                max_confianza=max_conf
            )
            db.session.add(record)
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