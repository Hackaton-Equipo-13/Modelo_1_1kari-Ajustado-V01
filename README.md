# 📊 Modelo de Análisis de Sentimiento Multilingüe

> **Detecta automáticamente si un comentario es positivo o negativo — en inglés, portugues y español — con alta confianza y preparado para integración en entornos empresariales.**

---

## 📌 1. Breve Descripción

Este Modelo implementa un modelo de clasificación binaria de sentimiento (positivo/negativo) diseñado para analizar reseñas de clientes en servicios o productos. Utiliza un enfoque híbrido basado en reglas léxicas para preprocesamiento y etiquetado inicial, seguido por un modelo supervisado de **Regresión Logística** entrenado con representación **TF-IDF** y optimizado para manejar datos en **inglés, portugues y español**. El modelo final se exporta en formato **ONNX**, listo para integrarse en sistemas backend, incluyendo aplicaciones Java. 
NOTA: Este modelo se ajustó con base en el propuesto inicialmente a dos sentimientos (Positivo/Negativo) y tiene mejor precisión al no tener en cuenta celdas con información basura 

Ideal para equipos de experiencia del cliente, soporte técnico o monitoreo de reputación digital.

---

## ✨ 2. Características Generales

- **Clasificación binaria robusta**: Distingue entre sentimientos **positivos (1)** y **negativos (0)**.
- **Soporte multilingüe**: Entrenado con datos reales en inglés y ejemplos sintéticos en español.
- **Manejo inteligente de casos críticos**: Palabras como *"estafa"*, *"fraude"* o *"robado"* fuerzan clasificación negativa, incluso si hay términos positivos.
- **Detección de negaciones**: Reconoce construcciones como *"not good"* y ajusta la predicción.
- **Filtrado automático**: Descarta comentarios vacíos, incoherentes o con muy pocas palabras.
- **Salida con probabilidad**: Proporciona nivel de confianza para cada predicción.
- **Exportación a ONNX**: Modelo serializado para despliegue en entornos productivos (incluyendo Java con ONNX Runtime).
- **Reportes visuales automáticos**: Genera gráficos de distribución y archivos CSV con resultados.

---

## 🛠️ 3. Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| **Lenguaje base** | Python 3.x |
| **Entorno de desarrollo** | Google Colab / Jupyter Notebook |
| **Librerías principales** | `scikit-learn`, `pandas`, `numpy`, `re`, `matplotlib`, `seaborn` |
| **Vectorización de texto** | `TfidfVectorizer` (unigramas y bigramas) |
| **Modelo de ML** | Regresión Logística (`class_weight='balanced'`) |
| **Serialización** | `joblib` (intermedio), **ONNX** (producción) |
| **Conversión a ONNX** | `skl2onnx` |
| **Ejecución en Java** | [ONNX Runtime for Java](https://onnxruntime.ai/docs/tutorials/java/) |

---

## 🗂️ 4. Estructura del Proyecto

```
sentiment-analysis-multilingual/
├── amazon_only_reviews_04.csv          # Dataset original (inglés)
├── amazon_final_cleaned.csv            # Dataset limpio y etiquetado
├── BrandPulse.onnx                     # Modelo final en formato ONNX
├── historial_sentimientos.csv          # Registro acumulativo de predicciones
├── distribucion_binaria_sentimientos.png
├── reporte_grafico_binario.png         # Gráficos generados
├── resultado_batch_con_probabilidad.csv # Salida en lote
└── Modelo brandPULSE.ipynb             # Notebook principal (entrenamiento y exportación)
```

---

## ⚙️ 5. Instalación y Uso

### Requisitos previos
- Acceso a Google Colab (o entorno Python local con `pip`)
- Google Drive vinculado (para cargar el dataset original)

### Pasos para ejecutar

1. **Montar Drive y clonar el notebook**  
   Ejecutar las primeras celdas del notebook para montar Google Drive.

2. **Instalar dependencias**  
   ```bash
   pip install scikit-learn skl2onnx onnxruntime onnx pandas matplotlib seaborn
   ```

3. **Ejecutar todo el notebook**  
   Secuencialmente, para:
   - Cargar y limpiar datos
   - Etiquetar con reglas léxicas
   - Balancear clases
   - Entrenar el modelo
   - Evaluar métricas
   - Exportar a ONNX

4. **Uso interactivo (opcional)**  
   Al final del notebook se incluyen funciones para:
   - Predecir un texto único: `procesar_predicciones(pipeline, "Me encantó el servicio!")`
   - Procesar un archivo CSV: `procesar_predicciones(pipeline, "comentarios.csv", es_archivo=True)`

5. **Descargar el modelo ONNX**  
   El archivo `BrandPulse.onnx` se genera automáticamente y puede descargarse desde Colab o Drive.

---

## 📈 6. Desempeño del Modelo

Después del balanceo de clases y entrenamiento con datos multilingües, el modelo obtuvo las siguientes métricas en el conjunto de prueba:

| Métrica       | Valor   |
|---------------|---------|
| **Accuracy**  | ~0.96   |
| **Precision** | ~0.96   |
| **Recall**    | ~0.96   |
| **F1-Score**  | ~0.96   |

> ✅ El uso de `class_weight='balanced'` y el enriquecimiento con ejemplos en español mejora significativamente la generalización y reduce el sesgo por desbalance.

La matriz de confusión muestra baja tasa de falsos negativos, crucial para no ignorar críticas graves.

---

## 🔗 7. Integración con Java

El modelo se exporta a **ONNX** (Open Neural Network Exchange), un formato estándar e interoperable. Para integrarlo en una aplicación Java:

1. **Agregar ONNX Runtime para Java** en `pom.xml`:
   ```xml
   <dependency>
     <groupId>com.microsoft.onnxruntime</groupId>
     <artifactId>onnxruntime</artifactId>
     <version>1.18.0</version>
   </dependency>
   ```

2. **Cargar y ejecutar el modelo**:
   ```java
   OrtEnvironment env = OrtEnvironment.getEnvironment();
   OrtSession session = env.createSession("sentiment_binary_multilingual.onnx", new OrtSession.SessionOptions());
   OnnxTensor input = OnnxTensor.createTensor(env, new String[][]{{"excelente servicio"}});
   OrtSession.Result result = session.run(Collections.singletonMap("text_input", input));
   ```

3. **Interpretar salida**:  
   - `result.get(0)` → etiqueta predicha (`0` o `1`)  
   - `result.get(1)` → probabilidades por clase

> 📌 El modelo espera una entrada de tipo **StringTensor de forma [1, 1]**.

---

## 🌐 Conclusión

Este sistema combina simplicidad, interpretabilidad y preparación para producción. Aunque basado en técnicas clásicas de NLP (TF-IDF + LR), su diseño cuidadoso —con manejo de negaciones, palabras críticas y datos multilingües— lo hace altamente efectivo para casos reales de feedback de usuarios. Su exportación a ONNX lo convierte en un componente reutilizable en arquitecturas monolíticas o microservicios, especialmente en entornos Java empresariales.

**¿Próximos pasos?**  
- Reentrenamiento continuo con feedback real  
- Extensión a más idiomas  
- API REST con FastAPI o Spring Boot  
- Monitoreo de drift de datos en producción

> 💡 **Listo para escalar. Listo para la empresa.**
