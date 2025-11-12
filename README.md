# Sentiment Analysis con RNN y BERT (PyTorch + TorchText)

Este proyecto implementa un sistema de **análisis de sentimientos** utilizando dos enfoques principales:

1. Una **RNN (GRU)** entrenada con embeddings preentrenados de **GloVe**.
2. Un modelo **BERT** combinado con una capa RNN para mejorar el rendimiento.

Se utiliza el conjunto de datos **IMDB** disponible en `torchtext.datasets`.

---

## 🚀 Características aprendidas

Durante este proyecto se aprende a:

- Usar **TorchText** para preparar datos de texto (tokenización, vocabulario y dataloaders).
- Implementar una **RNN personalizada** con embeddings GloVe.
- Cargar y usar **embeddings preentrenados**.
- Incorporar **BERT** de `transformers` como extractor de características.
- Congelar parámetros de BERT para transfer learning.
- Entrenar modelos con **PyTorch** y evaluar su rendimiento.
- Tokenizar texto tanto con **spaCy** como con **BERT Tokenizer**.
- Hacer predicciones con nuevas oraciones.

---

## 🧩 Estructura del código

El proyecto sigue esta secuencia lógica:

1. **Preparación del texto**
   - Tokenización con spaCy o BERT Tokenizer.
   - Creación de campos `TEXT` y `LABEL` con TorchText.

2. **Carga del dataset**
   - Uso del dataset `IMDB` proporcionado por `torchtext.datasets`.

3. **Modelo RNN**
   - Definición de una red GRU bidireccional.
   - Uso de embeddings GloVe (`glove.6B.100d`).
   - Entrenamiento y evaluación.

4. **Modelo BERT + RNN**
   - Uso de `BertModel` preentrenado (`bert-base-uncased`).
   - Congelación de capas de BERT.
   - Paso de las embeddings BERT a una GRU.
   - Clasificación final mediante una capa fully-connected.

5. **Predicciones**
   - Ejemplo de inferencia con frases nuevas.

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
