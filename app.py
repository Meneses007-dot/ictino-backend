from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. CONFIGURACIÓN DE LA BASE DE DATOS
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///proyecto.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==========================================
# 2. MODELOS DE LA BASE DE DATOS (TABLAS)
# ==========================================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='user') # 'user' o 'admin'
    
    detections = db.relationship('DetectionRecord', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class DetectionRecord(db.Model):
    __tablename__ = 'detection_records'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())
    results_json = db.Column(db.JSON, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

# ==========================================
# 3. VARIABLES GLOBALES Y CONFIGURACIÓN DE YOLO
# ==========================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
WEIGHTS_PATH  = os.path.join(BASE_DIR, "weights", "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO(WEIGHTS_PATH)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# 4. RUTAS DE ENTRADA Y AUTENTICACIÓN
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "Faltan usuario o contraseña"}), 400
        
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "El nombre de usuario ya está registrado"}), 400

    new_user = User(username=data["username"])
    new_user.set_password(data["password"])
    
    if data.get("role") in ["user", "admin"]:
        new_user.role = data["role"]

    db.session.add(new_user)
    db.session.commit()
    return jsonify({"success": True, "message": "Usuario registrado exitosamente"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "Faltan credenciales"}), 400

    user = User.query.filter_by(username=data["username"]).first()
    if not user or not user.check_password(data["password"]):
        return jsonify({"error": "Usuario o contraseña incorrectos"}), 401

    return jsonify({
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        }
    })

# ==========================================
# 5. PROCESAMIENTO CON YOLO Y PERSISTENCIA
# ==========================================
@app.route("/detect", methods=["POST"])
def detect():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "No se proporcionó el ID de usuario"}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Usuario no válido"}), 404

    if "image" not in request.files:
        return jsonify({"error": "No se subió ninguna imagen"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Formato no permitido. Usa PNG, JPG o WEBP"}), 400

    ext         = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    input_path  = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.save(input_path)
        results = model(input_path)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "confidence": round(float(box.conf[0]), 4),
            "class_id":   int(box.cls[0]),
            "box":        [round(float(x), 2) for x in box.xyxy[0].tolist()]
        })

    new_record = DetectionRecord(results_json=detections, user_id=user.id)
    db.session.add(new_record)
    db.session.commit()

    return jsonify({
        "success":    True,
        "record_id":  new_record.id,
        "detections": detections
    })

# ==========================================
# 6. GESTIÓN DEL HISTORIAL (VER Y BORRAR)
# ==========================================
@app.route("/history", methods=["GET"])
def get_history():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "No autorizado"}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Usuario no encontrado"}), 404

    if user.role == "admin":
        records = DetectionRecord.query.order_by(DetectionRecord.date.desc()).all()
    else:
        records = DetectionRecord.query.filter_by(user_id=user.id).order_by(DetectionRecord.date.desc()).all()

    output = []
    for r in records:
        output.append({
            "id": r.id,
            "date": r.date.strftime("%Y-%m-%d %H:%M:%S"),
            "results": r.results_json,
            "owner_username": r.owner.username
        })

    return jsonify({"success": True, "history": output})


@app.route("/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    data = request.get_json()
    user_id = data.get("user_id") if data else None
    
    if not user_id:
        return jsonify({"error": "No autorizado"}), 401

    user = User.query.get(user_id)
    record = DetectionRecord.query.get(record_id)

    if not record:
        return jsonify({"error": "El registro no existe"}), 404

    if record.user_id == user.id or user.role == "admin":
        db.session.delete(record)
        db.session.commit()
        return jsonify({"success": True, "message": "Registro eliminado"}), 200
    
    return jsonify({"error": "No tienes permisos para borrar este registro"}), 403

# ==========================================
# 7. DISPARADOR DEL SERVIDOR LOCAL
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)