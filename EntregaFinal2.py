from joblib import load
import pandas as pd
import streamlit as st


st.write("Inicio de la carga del modelo...")
# Cargar el modelo
modelo_cargado = load('modelo_bootstrap.joblib')

try:
    modelo_cargado = load('./modelo_bootstrap.joblib')
    st.write("Modelo cargado con éxito.")
except FileNotFoundError as e:
    st.error(f"Error: Archivo no encontrado - {e}")
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")

st.title("Predicción de Demencia")

caracteristicas_seleccionadas = ['Día', 'Año', 'Comuna', 'Rest793', 'Rest786', 'Rest779', 'Rest772',
       'Rest765', 'Limón2', 'LLave2', 'Puerta2', 'LetraP', 'Animales',
       'Avenida', 'Caldera', 'Copiapó', 'PresidenteEEUU',
       'PresidenteAsesinadoEEUU', 'Instrucción2', 'Instrucción3', 'Palabras',
       'Repetiroración1', 'Repetiroración2', 'Cocodrilo', 'Monarquía',
       'Palabras2', 'Cubo', 'Números', 'Puntero', '@8Puntos', 'Miguel2',
       'González2', 'Imperial2', '@682', 'Caldera2']

# Crear una barra lateral con opciones de selección para cada variable
respuestas_usuario = {}
with st.form("formulario_prediccion"):
    # Crear un diccionario para almacenar las respuestas del usuario
    respuestas_usuario = {}

    # Crear una barra lateral con opciones de selección para cada variable
    for caracteristica in caracteristicas_seleccionadas:
        respuestas_usuario[caracteristica] = st.radio(f"¿{caracteristica}?", ['Sí', 'No'])

    # Añadir un botón de envío al formulario
    submit_button = st.form_submit_button("Realizar Predicción")

# Realizar la predicción después de enviar el formulario
if submit_button:
    # Convertir respuestas del usuario a formato binario (Sí=1, No=0)
    respuestas_binarias = {caracteristica: 1 if respuesta == 'Sí' else 0 for caracteristica, respuesta in respuestas_usuario.items()}

    # Crear un DataFrame con las respuestas del usuario
    datos_usuario = pd.DataFrame([respuestas_binarias], columns=caracteristicas_seleccionadas)

    # Realizar la predicción utilizando el modelo cargado
    prediccion = modelo_cargado.predict(datos_usuario)

    # Traducir la etiqueta de la predicción
    etiquetas = {0: 'No', 1: 'Probable', 2: 'Posible'}
    resultado_prediccion = etiquetas[prediccion[0]]

    # Mostrar el resultado
    st.success(f"¡La predicción es: {resultado_prediccion}!")
