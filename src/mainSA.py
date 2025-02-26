import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
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

# Asegurarse de que los resultados sean reproducibles
#random.seed(42)
#np.random.seed(42)

# Cargar modelo de idioma inglés de Spacy
nlp = spacy.load("es_core_web_sm")

#nlp = spacy.load("C:/users/nancy/documents/proylucy/lucy/lib/site-packages/en_core_web_sm")


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


#def generar_combinacion_aleatoria():
 #   combinacion = [1, 0, 1, 0, random.choice([0, 1, 2, 3, 4]), random.choice([0, 1]), random.choice([0, 1, 2, 3])]
  #  return combinacion

#C0
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
def generar_combinacion_aleatoria():
    combinacion = [0, random.choice([0, 1]), random.choice([0, 1]), 0, 0, random.choice([0, 1]), random.choice([0, 1, 2, 3])]
    return combinacion

#C4
# [0, 1, 1, 0, a, b, c]
#def generar_combinacion_aleatoria():
 #   combinacion = [0, 1, 1, 0, random.choice([0, 1, 2, 3, 4]), random.choice([0, 1]), random.choice([0, 1, 2, 3])]
  #  return combinacion

#C5
#[a, b, 1, 0, c, d, e]
#def generar_combinacion_aleatoria():
 #   combinacion = [random.choice([0, 1]), random.choice([0, 1]), 1, 0, random.choice([0, 1, 2, 3, 4]), random.choice([0, 1]), random.choice([0, 1, 2, 3])]
  #  return combinacion
    

# Cargar los datos
datos = pd.read_csv('data/Twitter2_TRAIN.csv', encoding='iso-8859-1')

#datos.rename(columns={'ï»¿clean_text': 'clean_text'}, inplace=True)


# Convertir la columna 'clean_text' a cadenas de texto
#datos['clean_text'] = datos['clean_text'].astype(str)

# Eliminar filas con NaN en la columna 'category'
#datos = datos.dropna(subset=['category'])

print(datos)

# Función para aplicar técnicas y evaluar
def aplicar_tecnicas_y_evaluar(combinacion, datos):
    print("\n--- Aplicar técnicas y evaluar ---")
    print("Combinación aplicada:", combinacion)

    # Separar los datos en características y etiquetas
    #texto = datos['clean_text']
    #etiquetas = datos['category']

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
        texto_vectores = represent_text_bow(texto.apply(lambda x: ' '.join(x)))
        print("Representación de texto con Bag of Words aplicada.")
    else:
        texto_vectores = represent_text_tfidf(texto.apply(lambda x: ' '.join(x)))
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

    # Validación cruzada
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    resultados = cross_val_score(modelo, texto_vectores, etiquetas, cv=kfold, scoring='accuracy')
    precision_promedio = np.mean(resultados)

    print("Resultados de la validación cruzada:", resultados)
    print("Precisión media:", resultados.mean())
    print("Desviación estándar:", resultados.std())

    return precision_promedio

# Función para calcular la distancia euclidiana entre dos combinaciones
def calcular_distancia_euclidiana(combinacion1, combinacion2):
    return np.sqrt(np.sum((np.array(combinacion1) - np.array(combinacion2)) ** 2))

# Simulated annealing con visualización
import numpy as np

def simulated_annealing(funcion_objetivo, datos, temperatura_inicial=1000, factor_enfriamiento=0.20, temperatura_final=1e-10):
    combinacion_inicial = generar_combinacion_aleatoria()  # Genera una combinación inicial aleatoria
    combinacion_actual = combinacion_inicial  # Establece la combinación actual como la inicial
    valor_inicial = funcion_objetivo(combinacion_actual, datos)  # Calcula el valor inicial de la función objetivo
    valor_actual = valor_inicial  # Establece el valor actual como el valor inicial
    mejor_combinacion = combinacion_actual  # Inicialmente, la mejor combinación es la inicial
    mejor_valor = valor_actual  # Inicialmente, el mejor valor es el inicial
    temperatura = temperatura_inicial  # Establece la temperatura inicial
    valores_funcion_objetivo = [valor_actual]  # Lista para almacenar los valores de la función objetivo
    visualizacion_iteraciones_F1_C3_ejecucion_10 = []  # Lista para almacenar datos para la visualización
    distancia_total = 0  # Inicializa la distancia total a cero

    iteracion = 0  # Contador de iteraciones
    combinaciones = [combinacion_actual]
    aceptados = [True]  # Lista para almacenar si una combinación fue aceptada

    vecinos_historial = []  # Lista para almacenar los vecinos generados en cada iteración

    while temperatura > temperatura_final:  # Bucle principal que se ejecuta mientras la temperatura sea mayor que la final
        vecino1 = generar_combinacion_aleatoria()  # Genera el primer vecino aleatorio
        vecino2 = generar_combinacion_aleatoria()  # Genera el segundo vecino aleatorio
        vecino3 = generar_combinacion_aleatoria()  # Genera el tercer vecino aleatorio

        valor_vecino1 = funcion_objetivo(vecino1, datos)  # Calcula el valor de la función objetivo para el primer vecino
        valor_vecino2 = funcion_objetivo(vecino2, datos)  # Calcula el valor de la función objetivo para el segundo vecino
        valor_vecino3 = funcion_objetivo(vecino3, datos)  # Calcula el valor de la función objetivo para el tercer vecino

        mejor_vecino = vecino1  # Inicialmente, el mejor vecino es el primero
        mejor_valor_vecino = valor_vecino1  # Inicialmente, el mejor valor de vecino es el del primero
        if valor_vecino2 > mejor_valor_vecino:  # Si el valor del segundo vecino es mejor que el del primero
            mejor_vecino = vecino2  # El mejor vecino es el segundo
            mejor_valor_vecino = valor_vecino2  # El mejor valor de vecino es el del segundo
        if valor_vecino3 > mejor_valor_vecino:  # Si el valor del tercer vecino es mejor que el del segundo
            mejor_vecino = vecino3  # El mejor vecino es el tercero
            mejor_valor_vecino = valor_vecino3  # El mejor valor de vecino es el del tercero

        delta = mejor_valor_vecino - valor_actual  # Calcula la diferencia entre el mejor valor de vecino y el valor actual
        aceptado = delta > 0 or np.exp(delta / temperatura) > np.random.rand()  # Determina si se acepta el vecino según la condición de aceptación

        if aceptado:  # Si se acepta el vecino
            combinacion_anterior = combinacion_actual  # Guarda la combinación actual anterior
            combinacion_actual = mejor_vecino  # Actualiza la combinación actual al mejor vecino
            valor_actual = mejor_valor_vecino  # Actualiza el valor actual al mejor valor de vecino
            distancia_anterior = calcular_distancia_euclidiana(combinacion_anterior, combinacion_actual)  # Calcula la distancia euclidiana entre la combinación anterior y la actual
            distancia_total += distancia_anterior  # Suma la distancia calculada a la distancia total

            if mejor_valor_vecino > mejor_valor:  # Si el mejor valor de vecino es mayor que el mejor valor conocido
                mejor_combinacion = mejor_vecino  # Actualiza la mejor combinación
                mejor_valor = mejor_valor_vecino  # Actualiza el mejor valor

        comentario_aceptacion = "Sí" if aceptado else "No"  # Establece el comentario de aceptación
        visualizacion_iteraciones_F1_C3_ejecucion_10.append({
            "Iteración": iteracion,  # Número de iteración
            "Vecino 1": vecino1,  # Primer vecino generado
            "f(x) Vecino 1": valor_vecino1,  # Valor de la función objetivo para el primer vecino
            "Vecino 2": vecino2,  # Segundo vecino generado
            "f(x) Vecino 2": valor_vecino2,  # Valor de la función objetivo para el segundo vecino
            "Vecino 3": vecino3,  # Tercer vecino generado
            "f(x) Vecino 3": valor_vecino3,  # Valor de la función objetivo para el tercer vecino
            "Vecino Aceptado": mejor_vecino,  # Vecino aceptado
            "f(x) Vecino Aceptado": mejor_valor_vecino,  # Valor de la función objetivo para el vecino aceptado
            "Temperatura": temperatura,  # Temperatura actual
            "Se Acepta": comentario_aceptacion,  # Comentario de si se aceptó el vecino
            "Distancia Euclidiana": distancia_anterior if comentario_aceptacion == "Sí" else None  # Distancia euclidiana si se aceptó el vecino
        })

        combinaciones.append(combinacion_actual)  # Añade la combinación actual a la lista de combinaciones
        aceptados.append(aceptado)  # Añade el estado de aceptación a la lista

        vecinos_historial.append((vecino1, vecino2, vecino3))  # Almacena los vecinos generados en el historial

        temperatura *= factor_enfriamiento  # Disminuye la temperatura según el factor de enfriamiento
        valores_funcion_objetivo.append(valor_actual)  # Añade el valor actual de la función objetivo a la lista

        iteracion += 1  # Incrementa el contador de iteraciones

    with open('visualizacion_iteraciones_F1_C3_ejecucion_10.txt', 'w') as f:  # Abre un archivo para escribir los datos de visualización
        for registro in visualizacion_iteraciones_F1_C3_ejecucion_10:  # Recorre cada registro en los datos de visualización
            f.write(str(registro) + '\n')  # Escribe cada registro en el archivo

    print(f"Distancia total desde la combinación inicial hasta la mejor combinación hallada: {distancia_total:.2f}")  # Imprime la distancia total

    return mejor_combinacion, mejor_valor, valores_funcion_objetivo, combinacion_inicial, valor_inicial, visualizacion_iteraciones_F1_C3_ejecucion_10, combinaciones, aceptados, vecinos_historial  # Devuelve los resultados



# Visualización del espacio de soluciones y vecinos

# Llamada a la función simulated_annealing y obtención de los resultados
mejor_combinacion, mejor_valor, valores_funcion_objetivo, combinacion_inicial, valor_inicial, visualizacion_iteraciones_F1_C3_ejecucion_10, combinaciones, aceptados, vecinos_historial = simulated_annealing(aplicar_tecnicas_y_evaluar, datos)

print("\nMejor Solución Final:", mejor_combinacion)
print("Porcentaje de clasificación de la mejor solución:", mejor_valor)
print("\nPrimera Combinación Inicial:", combinacion_inicial)
print("Porcentaje de clasificación de la primera combinación:", valor_inicial)

##Generar matriz de confusión


# Graficar la evolución de la función objetivo
plt.figure(figsize=(14, 7))
plt.plot(valores_funcion_objetivo, label='Valor de la función objetivo')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función objetivo')
plt.title('Evolución de la función objetivo')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los valores de los vecinos y el aceptado en cada iteración
iteraciones = [registro["Iteración"] for registro in visualizacion_iteraciones_F1_C3_ejecucion_10]
valores_vecino1 = [registro["f(x) Vecino 1"] for registro in visualizacion_iteraciones_F1_C3_ejecucion_10]
valores_vecino2 = [registro["f(x) Vecino 2"] for registro in visualizacion_iteraciones_F1_C3_ejecucion_10]
valores_vecino3 = [registro["f(x) Vecino 3"] for registro in visualizacion_iteraciones_F1_C3_ejecucion_10]
valores_aceptados = [registro["f(x) Vecino Aceptado"] for registro in visualizacion_iteraciones_F1_C3_ejecucion_10]

plt.figure(figsize=(14, 7))
plt.plot(iteraciones, valores_vecino1, 'r', label='Valor Vecino 1')
plt.plot(iteraciones, valores_vecino2, 'g', label='Valor Vecino 2')
plt.plot(iteraciones, valores_vecino3, 'b', label='Valor Vecino 3')
plt.plot(iteraciones, valores_aceptados, 'k', label='Valor Aceptado')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de los vecinos') 
plt.title('Evolución de los valores de los vecinos y el aceptado')
plt.legend()
plt.grid(True)
plt.show()








