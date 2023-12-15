from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import plotly.express as px

def mostrarDatos(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df.head())

def eliminar_columnas(df, columnas_a_eliminar):
    nuevo_df = df.copy()
    nuevo_df.drop(columnas_a_eliminar, axis=1, inplace=True)
    return nuevo_df

def eliminar_filas_Demencia_nulos(df):
    column = df['Demencia']
    df_1 = df.dropna(subset=['Demencia'])
    df_1 = df_1.drop('Demencia', axis=1)
    
    imputer = SimpleImputer(strategy='mean')
    df_imputado = pd.DataFrame(imputer.fit_transform(df_1), columns=df_1.columns)
    df_imputado['Demencia'] = column

    df_imputado = df_imputado.dropna(subset=['Demencia'])
    return df_imputado

def mostrar_nulos(df):
    nulos_por_columna = df.isna().sum()
    info_nulos = pd.DataFrame({'Columna': nulos_por_columna.index, 'Valores NaN': nulos_por_columna.values})
    print(info_nulos)

def seleccionar_caracteristicas(X, y, num_caracteristicas=35):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y_encoded)

    sfm = SelectFromModel(clf, prefit=True, max_features=num_caracteristicas)
    caracteristicas_seleccionadas = X.columns[sfm.get_support()]

    # Obtener la importancia de cada característica
    importancias = clf.feature_importances_

    # Visualizar la importancia de las características
    plt.figure(figsize=(10, 6))
    plt.barh(caracteristicas_seleccionadas, importancias[sfm.get_support()])
    plt.xlabel('Importancia')
    plt.title('Importancia de las Características Seleccionadas')
    plt.show()
    
    
    return caracteristicas_seleccionadas

def bootstrap_predicciones_metrics(nuevo_df, columna_objetivo, num_bootstrap_samples=1105, max_depth=5, min_samples_leaf=5):
    all_predictions = []

    # Seleccionar características fuera del bucle
    X_original = nuevo_df.drop(columna_objetivo, axis=1)
    y_original = nuevo_df[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})
    caracteristicas_seleccionadas = seleccionar_caracteristicas(X_original, y_original)

    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = resample(nuevo_df, replace=True, random_state=42)

        X = bootstrap_sample.drop(columna_objetivo, axis=1)[caracteristicas_seleccionadas]
        y = bootstrap_sample[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})

        clf.fit(X, y)

        predictions = clf.predict(X_original[caracteristicas_seleccionadas])

        all_predictions.append(predictions)

    final_predictions = np.median(np.array(all_predictions), axis=0)

    # Calcular y mostrar el classification report
    target_names = ['No', 'Probable', 'Posible']
    report = classification_report(y_original, final_predictions, target_names=target_names)

    print(report)
   
def leave_one_out_cross_validation(X, y, clf):
    loo = LeaveOneOut()
    scores = []
    caracteristicas_seleccionadas_loo = seleccionar_caracteristicas(X, y)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Ajustar el clasificador al conjunto de entrenamiento
        clf.fit(X_train[caracteristicas_seleccionadas_loo], y_train)

        # Hacer predicciones en el conjunto de prueba
        predictions = clf.predict(X_test[caracteristicas_seleccionadas_loo])

        # Calcular la precisión y almacenarla en la lista de puntuaciones
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)

    # Calcular y mostrar el informe de clasificación final
    y_pred_final = clf.predict(X[caracteristicas_seleccionadas_loo])
    report = classification_report(y, y_pred_final)
    print("Informe de clasificación final:\n", report)

def mostrar_matriz_correlacion(df, caracteristicas_seleccionadas):
    # Filtrar el DataFrame original con las características seleccionadas y la etiqueta 'demencia'
    df_seleccionado = df[caracteristicas_seleccionadas]
    df[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})

    # Mostrar la matriz de correlación
    plt.figure(figsize=(12, 8))
    matriz_correlacion = df_seleccionado.corr()
    sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación con Características Seleccionadas y Etiqueta Demencia')
    plt.show()

def mostrarCorrelacion(df, columna_objetivo):
    df_2 = df_1_imputado.copy()
    X_original = df_2.drop(columna_objetivo, axis=1)
    y_original = df_2[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})
    caracteristicas_seleccionadas = seleccionar_caracteristicas(X_original, y_original)

    x2 = df_2.drop(columna_objetivo, axis=1)[caracteristicas_seleccionadas]
    y2 = df_2[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})

    x2['Demencia'] = y2

    correlation_matrix = x2.corr()
    correlation_with_label = correlation_matrix['Demencia'].drop('Demencia')

    # Dividir en correlaciones positivas y negativas
    positive_corr = correlation_with_label[correlation_with_label >= 0]
    negative_corr = correlation_with_label[correlation_with_label < 0]

    # Dividir las variables en dos grupos
    half_len = len(correlation_with_label) // 2
    group1 = correlation_with_label[:half_len]
    group2 = correlation_with_label[half_len:]

    # Crear dos gráficos de barras apiladas
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Gráfico 1
    axs[0].barh(group1.index, group1, color='lightgray')
    axs[0].barh(positive_corr.index.intersection(group1.index), positive_corr[positive_corr.index.intersection(group1.index)], color='green', label='Positivas')
    axs[0].barh(negative_corr.index.intersection(group1.index), negative_corr[negative_corr.index.intersection(group1.index)], color='red', label='Negativas')
    axs[0].set_xlabel('Correlación')
    axs[0].set_title('Correlación de Variables con la Etiqueta (Demencia) - Grupo 1')
    axs[0].legend()

    # Gráfico 2
    axs[1].barh(group2.index, group2, color='lightgray')
    axs[1].barh(positive_corr.index.intersection(group2.index), positive_corr[positive_corr.index.intersection(group2.index)], color='green', label='Positivas')
    axs[1].barh(negative_corr.index.intersection(group2.index), negative_corr[negative_corr.index.intersection(group2.index)], color='red', label='Negativas')
    axs[1].set_xlabel('Correlación')
    axs[1].set_title('Correlación de Variables con la Etiqueta (Demencia) - Grupo 2')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
   
filename = 'fonis-jbekios.sav'
df = pd.read_spss(filename)

# Columnas eliminando Demencia
columnas_eliminar1 = ['ID', 'Folio', 'Sexo', 'Edad', 'años_escolaridad', 'educacion', 'GDS_REC','GDS']

df_1 = eliminar_columnas(df, columnas_eliminar1)

df_1_imputado = eliminar_filas_Demencia_nulos(df_1)

columna_objetivo = 'Demencia'
mostrarDatos(df_1_imputado)
#predicciones_bootstrap = bootstrap_predicciones_metrics(df_1_imputado, columna_objetivo)

# Convertir las etiquetas reales a 0, 1, 2
etiquetas_reales = df_1_imputado[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})

#mostrar_matriz_correlacion(df_1_imputado, seleccionar_caracteristicas(df_1_imputado.drop(columna_objetivo, axis=1), df_1_imputado[columna_objetivo]))
# Utilizar la función leave_one_out_cross_validation
#leave_one_out_cross_validation(df_1_imputado.drop(columna_objetivo, axis=1), etiquetas_reales, DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5))
