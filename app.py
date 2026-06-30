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

resultado_estados = {
    0: 'alto',
    1: 'medio',
    2: 'bajo'
}

resultado_titulos = {
    0: 'Alto',
    1: 'Medio',
    2: 'Bajo'
}


def crear_recomendaciones(datos, prediccion_num):
    recomendaciones = []

    def agregar(titulo, detalle, nivel='media'):
        recomendaciones.append({
            'titulo': titulo,
            'detalle': detalle,
            'nivel': nivel
        })

    if datos['Avg_Daily_Usage_Hours'] >= 8:
        agregar(
            'Uso digital muy alto',
            'Reduce el uso en bloques concretos: desactiva notificaciones durante el estudio y reserva dos momentos del día para revisar redes.',
            'alta'
        )
    elif datos['Avg_Daily_Usage_Hours'] >= 6:
        agregar(
            'Uso digital elevado',
            'Prueba pausas sin pantalla entre clases o tareas, y define un límite diario realista para las aplicaciones que más tiempo consumen.',
            'media'
        )

    if datos['Sleep_Hours_Per_Night'] < 6:
        agregar(
            'Sueño insuficiente',
            'Prioriza una hora fija para dormir y aleja el celular antes de acostarte. Recuperar descanso suele mejorar energía, concentración y ánimo.',
            'alta'
        )
    elif datos['Sleep_Hours_Per_Night'] < 7:
        agregar(
            'Descanso mejorable',
            'Intenta sumar 30 minutos de sueño por noche durante la semana y evita estudiar con redes abiertas en la última hora del día.',
            'media'
        )

    if datos['Mental_Health_Score'] <= 4.5:
        agregar(
            'Bienestar emocional bajo',
            'Habla con una persona de confianza, tutor o servicio de bienestar estudiantil. Si el malestar se mantiene, busca apoyo profesional.',
            'alta'
        )
    elif datos['Mental_Health_Score'] <= 6:
        agregar(
            'Señales de desgaste',
            'Agenda descansos breves, actividad física ligera y espacios sin multitarea para bajar la carga mental durante el día.',
            'media'
        )

    if datos['Affects_Academic_Performance'] == 1:
        agregar(
            'Rendimiento afectado',
            'Divide las tareas grandes en metas pequeñas de 25 a 40 minutos y deja el celular fuera del escritorio mientras estudias.',
            'alta'
        )

    if prediccion_num == 0:
        agregar(
            'Plan de recuperación',
            'El resultado sugiere riesgo alto. Empieza con una meta de sueño, una meta de uso digital y una conversación de apoyo esta semana.',
            'alta'
        )
    elif prediccion_num == 1:
        agregar(
            'Seguimiento preventivo',
            'Tu estado puede cambiar según hábitos diarios. Revisa sueño, tiempo en redes y carga académica cada pocos días.',
            'media'
        )

    if not recomendaciones:
        agregar(
            'Buen punto de partida',
            'Mantén tus rutinas actuales y revisa tus hábitos cuando aumente la carga académica o el tiempo frente a pantallas.',
            'baja'
        )

    return recomendaciones


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
        'Avg_Daily_Usage_Hours': float(request.form['Avg_Daily_Usage_Hours']),
        'Most_Used_Platform': int(request.form['Most_Used_Platform']),
        'Sleep_Hours_Per_Night': float(request.form['Sleep_Hours_Per_Night']),
        'Mental_Health_Score': float(request.form['Mental_Health_Score']),
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
    prediccion_texto = resultado_titulos.get(prediccion_num, 'No definido')
    estado_resultado = resultado_estados.get(prediccion_num, 'medio')
    recomendaciones = crear_recomendaciones(datos, prediccion_num)

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
        prediccion=prediccion_texto,
        prediccion_num=prediccion_num,
        estado_resultado=estado_resultado,
        recomendaciones=recomendaciones,
        valores=datos
    )

# =========================
# EJECUTAR APP
# =========================

if __name__ == '__main__':

    init_db()

    app.run(debug=True)