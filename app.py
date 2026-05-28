from flask import Flask, render_template, request
import pandas as pd
import sqlite3
import joblib
import os

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

    if not os.path.exists('database.db'):

        conn = sqlite3.connect('database.db')

        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE respuestas (

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
        conn.close()

        print("Base de datos creada")

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

    datos = {

        'Age': int(request.form['Age']),
        'Gender': int(request.form['Gender']),
        'Academic_Level': int(request.form['Academic_Level']),
        'Avg_Daily_Usage_Hours': int(request.form['Avg_Daily_Usage_Hours']),
        'Most_Used_Platform': int(request.form['Most_Used_Platform']),
        'Sleep_Hours_Per_Night': int(request.form['Sleep_Hours_Per_Night']),
        'Mental_Health_Score': int(request.form['Mental_Health_Score']),
        'Affects_Academic_Performance': int(request.form['Affects_Academic_Performance'])

    }

    # DataFrame
    df = pd.DataFrame([datos])

    # ORDEN EXACTO DE COLUMNAS DEL ENTRENAMIENTO
    df = df[[
        'Age',
        'Gender',
        'Academic_Level',
        'Avg_Daily_Usage_Hours',
        'Most_Used_Platform',
        'Affects_Academic_Performance',
        'Sleep_Hours_Per_Night',
        'Mental_Health_Score'
    ]]

    # Escalar columnas numéricas
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predicción
    prediccion_num = int(modelo.predict(df)[0])

    # Texto resultado
    if prediccion_num == 0:
        prediccion_texto = "Negativo"

    elif prediccion_num == 1:
        prediccion_texto = "Neutral"

    else:
        prediccion_texto = "Positivo"

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
        prediccion_num

    ))

    conn.commit()
    conn.close()

    return render_template(
        'index.html',
        prediccion=prediccion_texto
    )

# =========================
# EJECUTAR APP
# =========================

if __name__ == '__main__':

    init_db()

    app.run(debug=True)