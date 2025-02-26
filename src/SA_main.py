import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, PunktSentenceTokenizer, WordPunctTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import re
import spacy
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import os
import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score, make_scorer

# Cargamos el modelo de idioma español de Spacy
nlp = spacy.load("es_core_news_sm")


# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Listas de técnicas con números asignados
eliminar_puntuacion = [0, 1]  # 0: "Si", 1: "No"
conversion_de_texto = [0, 1]  # 0: "Minúscula", 1: "Mayúscula"
reduccion_de_texto = [0, 1]  # 0: "Stemming", 1: "Lematización"
eliminacion_de_ruido = [0, 1]  # 0: "Si", 1: "No"
tokenizacion_de_texto = [0, 1, 2, 3, 4]  # 0: "Word_tokenize", 1: "TreebankWordTokenizer", 2: "PunktTokenizer", 3: "WordPunctTokenizer", 4: "RegexpTokenizer", 5: "sent_tokenize"
representacion_de_texto = [0, 1]  # 0: "Bag of Words", 1: "TF-IDF"
modelos_de_clasificacion = [0, 1, 2, 3]  # 0: "Naive Bayes", 1: "SVM", 2: "Árbol de decisión", 3: "KNN"



# Funciones de preprocesamiento de texto
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])

def convert_case(text, lower=True):
    return text.lower() if lower else text.upper()


def reduce_text(text, use_stemming=True):
    if use_stemming:
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])
    else:
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def eliminar_ruido(texto):
    return re.sub(r'[^a-zA-Z\s]', '', texto)

def tokenize_text(text, method=0):
    if method == 0:
        return word_tokenize(text)
    elif method == 1:
        tokenizer = TreebankWordTokenizer()
        return tokenizer.tokenize(text)
    elif method == 2:
        tokenizer = PunktSentenceTokenizer()
        return tokenizer.tokenize(text)
    elif method == 3:
        tokenizer = WordPunctTokenizer()
        return tokenizer.tokenize(text)
    elif method == 4:
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(text)

def represent_text_bow(texto):
    vectorizer = CountVectorizer()
    texto_vectores = vectorizer.fit_transform(texto)
    return texto_vectores

def represent_text_tfidf(textos):
    vectorizer = TfidfVectorizer()
    texto_vectores = vectorizer.fit_transform(textos)
    return texto_vectores


# Función para generar una combinación aleatoria de técnicas respetando las tareas
#C0
#E0
# [a, b, c, d, e, f, g]
#def generar_combinacion_aleatoria():
 #   combinacion = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1, 2, 3, 4]), random.choice([0, 1]), random.choice([0, 1, 2, 3])]
  #  return combinacion

#C1
# [0, 0,  a, b, 1, 0, c]
#def generar_combinacion_aleatoria():
 #   combinacion = [0, 0, random.choice([0, 1]), random.choice([0, 1]), 1, 0, random.choice([0, 1, 2, 3])]
  #  return combinacion

#C2
# [0, 0, 1, a, b, 1, c]
#def generar_combinacion_aleatoria():
 #   combinacion = [0, 0, 1, random.choice([0, 1]), random.choice([0, 1, 2, 3, 4]), 1, random.choice([0, 1, 2, 3])]
  #  return combinacion

#C3
# [0, a, b, 0, 0, c, d]
#def generar_combinacion_aleatoria():
 #   combinacion = [0, random.choice([0, 1]), random.choice([0, 1]), 0, 0, random.choice([0, 1]), random.choice([0, 1, 2, 3])]
  #  return combinacion

#C4
# [0, 1, 1, 0, a, b, c]
#def generar_combinacion_aleatoria():
 #   combinacion = [0, 1, 1, 0, random.choice([0, 1, 2, 3, 4]), random.choice([0, 1]), random.choice([0, 1, 2, 3])]
  #  return combinacion

#C5
#[a, b, 1, 0, c, d, e]
def generar_combinacion_aleatoria():
    combinacion = [random.choice([0, 1]), random.choice([0, 1]), 1, 0, random.choice([0, 1, 2, 3, 4]), random.choice([0, 1]), random.choice([0, 1, 2, 3])]
    return combinacion
    

# Cargar los datos
datos = pd.read_csv('data/Twitter2_TRAIN.csv', encoding='iso-8859-1')
print(datos)

# Función para aplicar técnicas y evaluar
def aplicar_tecnicas_y_evaluar(combinacion, datos):
    print("\n--- Aplicar técnicas y evaluar ---")
    print("Combinación aplicada:", combinacion)

    # Separamos los datos en características y etiquetas
    texto = datos['text']
    etiquetas = datos['sentiment']

    # Aplicar técnicas de preprocesamiento
    if combinacion[0] == 0:
        texto = texto.apply(remove_punctuation)
        print("Eliminación de puntuación aplicada.")
    if combinacion[0] == 1:
        texto = texto  # No se aplica remove_punctuation
        print("Eliminación de puntuación NO aplicada.")
    if combinacion[1] == 0:
        texto = texto.apply(lambda x: convert_case(x, lower=True))
        print("Conversión de texto a minúsculas aplicada.")
    else:
        texto = texto.apply(lambda x: convert_case(x, lower=False))
        print("Conversión de texto a mayúsculas aplicada.")
    if combinacion[2] == 0:
        texto = texto.apply(lambda x: reduce_text(x, use_stemming=True))
        print("Stemming aplicado.")
    else:
        texto = texto.apply(lambda x: reduce_text(x, use_stemming=False))
        print("Lematización aplicada.")
    if combinacion[3] == 0:
        texto = texto.apply(eliminar_ruido)
        print("Eliminación de ruido aplicada.")
    if combinacion[3] == 1:
        texto = texto  # No se aplica eliminar_ruido
        print("Eliminación de ruido NO aplicada.")

    texto = texto.apply(lambda x: tokenize_text(x, method=combinacion[4]))
    print(f"Tokenización de texto aplicada con método {combinacion[4]}.")

    # Representación de texto
    if combinacion[5] == 0:
        vectorizador = CountVectorizer()  # Creación del vectorizador
        print("Representación de texto con Bag of Words aplicada.")
    else:
        vectorizador = TfidfVectorizer()  # Creación del vectorizador
        print("Representación de texto con TF-IDF aplicada.")

    # Selección del modelo de clasificación
    if combinacion[6] == 0:
        modelo = MultinomialNB()
        print("Modelo de clasificación: Naive Bayes.")
    elif combinacion[6] == 1:
        modelo = SVC(kernel='linear')
        print("Modelo de clasificación: SVM.")
    elif combinacion[6] == 2:
        modelo = DecisionTreeClassifier()
        print("Modelo de clasificación: Árbol de decisión.")
    elif combinacion[6] == 3:
        modelo = KNeighborsClassifier()
        print("Modelo de clasificación: KNN.")

    # Cargamos las particiones predefinidas
    particiones = pd.read_csv('Particiones2.csv')   

    # Diccionario para almacenar los resultados
    resultados = {"Pliegue": [], "Accuracy": []}

    # Iteramos sobre cada pliegue único
    for pliegue in particiones['Pliegue'].unique():
        # Obtenemos los índices de entrenamiento y prueba para este pliegue
        train_idx = particiones[(particiones['Pliegue'] == pliegue) & (particiones['Tipo'] == 'train')]['Indice'].tolist()
        test_idx = particiones[(particiones['Pliegue'] == pliegue) & (particiones['Tipo'] == 'test')]['Indice'].tolist()

        # Separamos los datos en entrenamiento y prueba
        X_train_text = datos.iloc[train_idx]['text']
        y_train = datos.iloc[train_idx]['sentiment']
        X_test_text = datos.iloc[test_idx]['text']
        y_test = datos.iloc[test_idx]['sentiment']

        # Ajustamos el vectorizador con el conjunto de entrenamiento y transformar ambos conjuntos
        X_train = vectorizador.fit_transform(X_train_text)
        X_test = vectorizador.transform(X_test_text)

        # Entrenamos el modelo
        modelo.fit(X_train, y_train)

        # Predecimos en el conjunto de prueba
        y_pred = modelo.predict(X_test)

        # Calculamos la precisión
        accuracy = accuracy_score(y_test, y_pred)

        # Guardamos los resultados
        resultados["Pliegue"].append(pliegue)
        resultados["Accuracy"].append(accuracy)

        # Imprimimos los resultados del pliegue actual
        print(f"Pliegue {pliegue}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("-" * 50)

    # Convertimos los resultados a un DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Calculamos la precisión media y la desviación estándar
    precision_media = df_resultados['Accuracy'].mean()
    desviacion_estandar = df_resultados['Accuracy'].std()

    # Imprimimos la precisión media y la desviación estándar
    print(f"Precisión media: {precision_media:.4f}")
    print(f"Desviación estándar de la precisión: {desviacion_estandar:.4f}")

    # Retornamos los resultados y las métricas
    #return df_resultados, precision_media, desviacion_estandar
    return precision_media

# Función para calcular la distancia euclidiana entre dos combinaciones
def calcular_distancia_euclidiana(combinacion1, combinacion2):
    return np.sqrt(np.sum((np.array(combinacion1) - np.array(combinacion2)) ** 2))




# Simulated annealing con visualización
def simulated_annealing(funcion_objetivo, datos, temperatura_inicial=1000, factor_enfriamiento=0.20, temperatura_final=1e-10):
    combinacion_inicial = generar_combinacion_aleatoria()
    combinacion_actual = combinacion_inicial
    valor_inicial = funcion_objetivo(combinacion_actual, datos)
    valor_actual = valor_inicial
    mejor_combinacion = combinacion_actual
    mejor_valor = valor_actual
    temperatura = temperatura_inicial
    valores_funcion_objetivo = [valor_actual]
    visualizacion_iteraciones = []
    distancia_total = 0

    iteracion = 0
    combinaciones = [combinacion_actual]
    aceptados = [True]
    vecinos_historial = []

    while temperatura > temperatura_final:
        vecino1 = generar_combinacion_aleatoria()
        vecino2 = generar_combinacion_aleatoria()
        vecino3 = generar_combinacion_aleatoria()

        valor_vecino1 = funcion_objetivo(vecino1, datos)
        valor_vecino2 = funcion_objetivo(vecino2, datos)
        valor_vecino3 = funcion_objetivo(vecino3, datos)

        mejor_vecino = vecino1
        mejor_valor_vecino = valor_vecino1
        if valor_vecino2 > mejor_valor_vecino:
            mejor_vecino = vecino2
            mejor_valor_vecino = valor_vecino2
        if valor_vecino3 > mejor_valor_vecino:
            mejor_vecino = vecino3
            mejor_valor_vecino = valor_vecino3

        delta = mejor_valor_vecino - valor_actual
        aceptado = delta > 0 or np.exp(delta / temperatura) > np.random.rand()

        if aceptado:
            combinacion_anterior = combinacion_actual
            combinacion_actual = mejor_vecino
            valor_actual = mejor_valor_vecino
            distancia_anterior = calcular_distancia_euclidiana(combinacion_anterior, combinacion_actual)
            distancia_total += distancia_anterior

            if mejor_valor_vecino > mejor_valor:
                mejor_combinacion = mejor_vecino
                mejor_valor = mejor_valor_vecino

        comentario_aceptacion = "Sí" if aceptado else "No"
        visualizacion_iteraciones.append({
            "Iteración": iteracion,
            "Vecino 1": vecino1,
            "f(x) Vecino 1": valor_vecino1,
            "Vecino 2": vecino2,
            "f(x) Vecino 2": valor_vecino2,
            "Vecino 3": vecino3,
            "f(x) Vecino 3": valor_vecino3,
            "Vecino Aceptado": mejor_vecino,
            "f(x) Vecino Aceptado": mejor_valor_vecino,
            "Temperatura": temperatura,
            "Se Acepta": comentario_aceptacion,
            "Distancia Euclidiana": distancia_anterior if comentario_aceptacion == "Sí" else None
        })

        combinaciones.append(combinacion_actual)
        aceptados.append(aceptado)
        vecinos_historial.append((vecino1, vecino2, vecino3))

        temperatura *= factor_enfriamiento
        valores_funcion_objetivo.append(valor_actual)
        iteracion += 1

    return mejor_combinacion, mejor_valor, valores_funcion_objetivo, combinacion_inicial, valor_inicial, visualizacion_iteraciones, combinaciones, aceptados, vecinos_historial, distancia_total

# Listas para acumular los resultados de todas las ejecuciones
resultados_finales = []



for i in range(10):
    mejor_combinacion, mejor_valor, valores_funcion_objetivo, combinacion_inicial, valor_inicial, visualizacion_iteraciones, combinaciones, aceptados, vecinos_historial, distancia_total = simulated_annealing(aplicar_tecnicas_y_evaluar, datos)
    
    # Guardar los datos de visualización en un archivo CSV
    df_visualizacion = pd.DataFrame(visualizacion_iteraciones)
    df_visualizacion.to_csv(f'visualizacion_iteraciones_F0.80_E5_ejecucion_{i+1}.csv', index=False)

    print(f"Los datos de visualización se han guardado en 'visualizacion_iteraciones_F0.80_E5_ejecucion_{i+1}.csv'.")

    
    # Graficar la evolución de la función objetivo
    plt.figure(figsize=(14, 7))
    plt.plot(valores_funcion_objetivo, label='Value of the objective function')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness function value') 
    #plt.title('Evolución de la función objetivo')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'E5_Eje{i+1}.png')  # Guardar la figura con nombre único para cada iteración
    plt.close()  # Cerrar la figura para que no se muestre

    # Acumular resultados para imprimir al final
    resultados_finales.append(f"Iteración {i+1}:")
    resultados_finales.append(f"Distancia total desde la combinación inicial hasta la mejor combinación hallada: {distancia_total:.2f}")
    resultados_finales.append(f"Mejor Solución Final: {mejor_combinacion}")
    resultados_finales.append(f"Porcentaje de clasificación de la mejor solución: {mejor_valor}")
    resultados_finales.append(f"\nPrimera Combinación Inicial: {combinacion_inicial}")
    resultados_finales.append(f"Porcentaje de clasificación de la primera combinación: {valor_inicial}\n")

# Imprimir todos los resultados juntos al final
for resultado in resultados_finales:
    print(resultado)