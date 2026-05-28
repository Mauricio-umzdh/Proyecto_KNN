from flask import Flask, render_template, request
import pandas as pd
import sqlite3
import joblib

app = Flask(__name__)

# =========================
# CARGAR MODELO Y SCALER
# =========================

modelo = joblib.load('modelo_knn.pkl')
scaler = joblib.load('scaler.pkl')

# =========================
# COLUMNAS NUMÉRICAS
# =========================

numeric_cols = [
    'Age',
    'Avg_Daily_Usage_Hours',
    'Sleep_Hours_Per_Night',
    'Mental_Health_Score'
]

# =========================
# CREAR BASE DE DATOS
# =========================

def init_db():

    conn = sqlite3.connect('database.db')

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS respuestas (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        Age REAL,
        Gender INTEGER,
        Academic_Level INTEGER,
        Avg_Daily_Usage_Hours REAL,
        Most_Used_Platform INTEGER,
        Sleep_Hours_Per_Night REAL,
        Mental_Health_Score REAL,
        Affects_Academic_Performance INTEGER,

        prediccion INTEGER
    )
    ''')

    conn.commit()

    print("Base de datos creada correctamente")

    conn.close()

# =========================
# RUTA PRINCIPAL
# =========================

@app.route('/')
def index():

    return render_template('index.html')

# =========================
# PREDICCIÓN
# =========================

@app.route('/predecir', methods=['POST'])
def predecir():

    # Obtener datos del formulario
    datos = {

        'Age': float(request.form['Age']),
        'Gender': int(request.form['Gender']),
        'Academic_Level': int(request.form['Academic_Level']),
        'Avg_Daily_Usage_Hours': float(request.form['Avg_Daily_Usage_Hours']),
        'Most_Used_Platform': int(request.form['Most_Used_Platform']),
        'Sleep_Hours_Per_Night': float(request.form['Sleep_Hours_Per_Night']),
        'Mental_Health_Score': float(request.form['Mental_Health_Score']),
        'Affects_Academic_Performance': int(request.form['Affects_Academic_Performance'])

    }

    # Convertir datos a DataFrame
    df = pd.DataFrame([datos])

    # Escalar columnas numéricas
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Realizar predicción
    prediccion = modelo.predict(df)[0]

    # =========================
    # GUARDAR EN SQLITE
    # =========================

    conn = sqlite3.connect('database.db')

    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO respuestas (

        Age,
        Gender,
        Academic_Level,
        Avg_Daily_Usage_Hours,
        Most_Used_Platform,
        Sleep_Hours_Per_Night,
        Mental_Health_Score,
        Affects_Academic_Performance,
        prediccion

    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (

        datos['Age'],
        datos['Gender'],
        datos['Academic_Level'],
        datos['Avg_Daily_Usage_Hours'],
        datos['Most_Used_Platform'],
        datos['Sleep_Hours_Per_Night'],
        datos['Mental_Health_Score'],
        datos['Affects_Academic_Performance'],
        int(prediccion)

    ))

    conn.commit()

    conn.close()

    # Mostrar resultado
    return render_template(
        'index.html',
        prediccion=prediccion
    )

# =========================
# EJECUTAR APP
# =========================

if __name__ == '__main__':

    init_db()

    app.run(host='0.0.0.0', port=5000)