import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Configuración de la Base de Datos (Soporta SQLite local o Neon PostgreSQL vía variable de entorno)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ictino.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_ictino')

db = SQLAlchemy(app)

# ── MODELOS DE LA BASE DE DATOS ──

class User(db.Model):
    __tablename__ = 'usuarios'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)


class DetectionRecord(db.Model):
    __tablename__ = 'historial_detecciones'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=True)
    fecha = db.Column(db.String(50), nullable=False)
    latitud = db.Column(db.Float, nullable=True)
    longitud = db.Column(db.Float, nullable=True)
    max_confianza = db.Column(db.Float, nullable=False)


# ── RUTA DE LA INTERFAZ ──

@app.route('/')
def index():
    return render_template('index.html')


# ── RUTA 1: INICIAR SESIÓN (Corrección Problema 1 y 2) ──
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({"error": "Por favor ingresa usuario y contraseña."}), 400

        # Buscamos al usuario de forma segura
        user = db.session.execute(
            db.select(User).filter_by(username=username)
        ).scalar_one_or_none()

        if user and check_password_hash(user.password, password):
            # Solución Problema 1: user_id enviado directamente en la raíz para index.html (Línea 441)
            return jsonify({
                "user_id": user.id,
                "status": "success"
            }), 200
        
        return jsonify({"error": "Usuario o contraseña incorrectos."}), 401

    except Exception as e:
        print(f"Error en /login: {str(e)}")
        return jsonify({"error": "Error interno en el servidor"}), 500


# ── RUTA 2: REGISTRO DE USUARIOS (Corrección Problema 4) ──
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({"error": "Por favor ingresa usuario y contraseña."}), 400

        # Validamos si el usuario ya existe
        existing_user = db.session.execute(
            db.select(User).filter_by(username=username)
        ).scalar_one_or_none()

        if existing_user:
            return jsonify({"error": "El nombre de usuario ya está registrado."}), 400

        # Encriptamos la contraseña con Werkzeug
        hashed_password = generate_password_hash(password, method='scrypt')
        new_user = User(username=username, password=hashed_password)

        # Solución Problema 4: Control de fallos con Neon usando try/except y rollback
        try:
            db.session.add(new_user)
            db.session.commit()
            return jsonify({
                "status": "success",
                "message": "Usuario creado correctamente"
            }), 201
        except Exception as db_err:
            db.session.rollback()  # Evita dejar la sesión corrupta
            print(f"Error de BD en registro: {str(db_err)}")
            return jsonify({"error": "Error al guardar el usuario en la base de datos."}), 500

    except Exception as e:
        print(f"Error general en /register: {str(e)}")
        return jsonify({"error": "Error interno en el servidor"}), 500


# ── RUTA 3: DETECCIÓN DE IMÁGENES (Corrección Problema 2 y 4) ──
@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen"}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Archivo vacío"}), 400

        # Solución Problema 2: Casteo seguro de user_id de str a int, y uso de db.session.get()
        user_id_raw = request.form.get('user_id')
        user = None
        if user_id_raw:
            try:
                user_id = int(user_id_raw)
                user = db.session.get(User, user_id)  # API moderna de SQLAlchemy 2.x
            except (ValueError, TypeError):
                return jsonify({"error": "ID de usuario inválido"}), 400

        # ── AQUÍ CORRE TU MODELO DE IA (Ejemplo Simulado) ──
        detections = [
            {"box": [30, 45, 180, 240], "confidence": 0.92}
        ]

        # Si decides persistir la telemetría del análisis en tu base de datos:
        if user and detections:
            new_record = DetectionRecord(
                user_id=user.id,
                fecha="17/05/2026, 18:21:00",
                latitud=None,  # Extraído del EXIF por el frontend o procesado aquí
                longitud=None,
                max_confianza=max(d['confidence'] for d in detections)
            )
            # Solución Problema 4: Rollback seguro si la transacción a Neon se interrumpe
            try:
                db.session.add(new_record)
                db.session.commit()
            except Exception as db_err:
                db.session.rollback()
                print(f"Error al guardar registro de detección: {str(db_err)}")
                # Nota: No bloqueamos la respuesta del análisis aunque la BD falle
        
        return jsonify({"detections": detections}), 200

    except Exception as e:
        print(f"Error en /detect: {str(e)}")
        return jsonify({"error": f"Error en el análisis: {str(e)}"}), 500


# ── RUTA 4: HISTORIAL (Corrección Problema 2) ──
@app.route('/history', methods=['GET'])
def history():
    try:
        user_id_raw = request.args.get('user_id')
        if not user_id_raw:
            return jsonify({"error": "Falta el parámetro user_id"}), 400

        # Solución Problema 2: Validación e int() + consulta moderna
        try:
            user_id = int(user_id_raw)
        except (ValueError, TypeError):
            return jsonify({"error": "user_id debe ser un entero válido"}), 400

        user = db.session.get(User, user_id)
        if not user:
            return jsonify({"error": "Usuario no encontrado"}), 404

        # Consulta estructurada en SQLAlchemy 2.0
        records = db.session.execute(
            db.select(DetectionRecord).filter_by(user_id=user_id)
        ).scalars().all()

        return jsonify({
            "status": "success",
            "history": [
                {
                    "id": r.id,
                    "fecha": r.fecha,
                    "lat": r.latitud,
                    "lng": r.longitud,
                    "max_conf": r.max_confianza
                } for r in records
            ]
        }), 200

    except Exception as e:
        print(f"Error en /history: {str(e)}")
        return jsonify({"error": "Error al recuperar el historial"}), 500


# ── RUTA 5: ELIMINAR REGISTROS (Corrección Problema 4) ──
@app.route('/delete_record', methods=['POST'])
def delete_record():
    try:
        data = request.get_json()
        record_id = data.get('record_id')

        record = db.session.get(DetectionRecord, record_id)
        if not record:
            return jsonify({"error": "Registro no encontrado"}), 404

        # Solución Problema 4: Control de errores transaccionales
        try:
            db.session.delete(record)
            db.session.commit()
            return jsonify({"status": "success", "message": "Registro eliminado"}), 200
        except Exception as db_err:
            db.session.rollback()
            print(f"Error al borrar de la BD: {str(db_err)}")
            return jsonify({"error": "No se pudo eliminar de la base de datos"}), 500

    except Exception as e:
        print(f"Error en /delete_record: {str(e)}")
        return jsonify({"error": "Error interno"}), 500


# ── CONTROL DE ARRANQUE E INICIALIZACIÓN ──

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # Lee el puerto dinámico de Render, si no existe (local), usa el 5000
    port = int(os.environ.get('PORT', 5000))
    
    # En producción, debug debe ser False para evitar vulnerabilidades
    app.run(host="0.0.0.0", port=port, debug=False)