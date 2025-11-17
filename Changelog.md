# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.4.0] - 2025-11-17

### 🚀 Agregado
- **FASE 13: SHAP Analysis Exhaustivo Corregido**
  - Implementación completa de 6 tipos de visualizaciones SHAP
  - Manejo robusto de diferentes formatos de SHAP values
  - Análisis de coherencia clínica automatizado
  - Exportación de estadísticas SHAP a CSV
  - Try-catch para visualizaciones opcionales

### 🔧 Corregido
- **FASE 12: Selección de Mejor Modelo**
  - Corregido error `NameError: name 'comparison' is not defined`
  - Implementado construcción del DataFrame de comparación antes de usar
  - Agregado sistema de composite score ponderado (AUC 40%, F1 40%, Acc 20%)
  - Implementada penalización por overfitting (gap > 5%)
  - Corregidas visualizaciones de comparación de modelos

- **FASE 13: SHAP Analysis**
  - Corregido error `NameError: name 'shap_values_to_use' is not defined`
  - Corregido `TypeError` en formato de strings con pandas.Index
  - Reemplazada API privada `_waterfall.waterfall_legacy` por API pública
  - Implementado manejo robusto de `expected_value` (array vs escalar)
  - Corregida conversión de pandas.Index a lista de strings
  - Agregado manejo de SHAP values multidimensionales (lista, 3D, 2D)
  - Corregido acceso a y_test con conversión a numpy array

- **FASE 7-11: Variables para FASE 12**
  - Agregadas variables individuales en cada fase de modelado:
    - `lr_train_acc`, `lr_test_acc`, `lr_f1`, `lr_auc`
    - `rf_train_acc`, `rf_test_acc`, `rf_f1`, `rf_auc`
    - `xgb_train_acc`, `xgb_test_acc`, `xgb_f1`, `xgb_auc`
    - `nn_train_acc`, `nn_test_acc`, `nn_f1`, `nn_auc`
    - `ensemble_train_acc`, `ensemble_test_acc`, `ensemble_f1`, `ensemble_auc`

### 📚 Documentación
- Agregado **README.md** completo para GitHub con:
  - Descripción exhaustiva del proyecto
  - Badges de tecnologías
  - Stack completo MLOps
  - Instrucciones de instalación y uso
  - Pipeline MLOps documentado
  - Resultados y comparación de modelos
  
- Creados **7 documentos de mejoras**:
  - `CORRECCIONES_NOTEBOOK_LAB_U3.md` - Guía técnica completa
  - `CODIGOS_COPIAR_PEGAR_NOTEBOOK.md` - Código listo para usar
  - `ANALISIS_COMPLETO_Y_MEJORAS.md` - Resumen ejecutivo
  - `CHEAT_SHEET_CORRECCIONES.md` - Referencia rápida
  - `INDICE_MAESTRO.md` - Navegación de documentos
  - `FASE_13_SHAP_CORREGIDA_DEFINITIVA.py` - SHAP corregida
  - `EXPLICACION_CORRECCIONES_FASE_13.md` - Detalles técnicos

### ⚡ Mejorado
- **Selección de Modelos**: Implementado sistema de scoring compuesto que prioriza métricas médicamente relevantes
- **Robustez del Código**: Agregadas validaciones y try-catch en puntos críticos
- **Mensajes Informativos**: Mejorado logging con información de shapes y tipos
- **Manejo de Errores**: Implementado fallback graceful para visualizaciones opcionales

---

## [3.3.1] - 2025-11-16

### 🔧 Corregido
- **FASE 6**: Agregar `y_train_final` e `y_test_final` para consistencia
- Variables de target separadas para evitar confusión entre scaled y original

---

## [3.3.0] - 2025-11-16

### 🚀 Agregado
- **FASE 15**: Evidently AI - Reportes automáticos de clasificación
- Sistema de try-catch robusto para Evidently AI
- Fallback graceful si Evidently no está disponible

---

## [3.2.0] - 2025-11-16

### 🚀 Agregado
- **FASE 14**: Deepchecks - Validación exhaustiva de modelos
- 30+ checks automáticos de integridad y performance
- Generación de reporte HTML interactivo

---

## [3.1.0] - 2025-11-16

### 🚀 Agregado
- **FASE 13**: SHAP Analysis - Interpretabilidad completa
- 6 tipos de visualizaciones SHAP
- Análisis cuantitativo de SHAP values
- Validación de coherencia clínica

---

## [3.0.0] - 2025-11-16

### 🚀 Agregado
- **FASE 12**: Comparación Final de Modelos
- Sistema de selección del mejor modelo
- Visualizaciones de comparación
- Guardado del mejor modelo con joblib

### ⚡ Mejorado
- Estructura modular de 16 fases
- Sistema de tracking completo con MLflow
- Integración fluida entre todas las fases

---

## [2.9.0] - 2025-11-16

### 🚀 Agregado
- **FASE 5.9**: Implementación manual de PCA/t-SNE/UMAP
- Más robusto que Yellowbrick para visualizaciones multivariadas
- Manejo correcto de dimensionalidad

---

## [2.8.0] - 2025-11-16

### 🔧 Corregido
- **FASE 5.9**: Agregar imputación de NaN para visualizaciones multivariadas
- SimpleImputer con estrategia 'mean' para PCA/t-SNE/UMAP

---

## [2.7.0] - 2025-11-15

### 🔧 Corregido
- **FASE 5.6**: Corregir ranking features
- Workaround para pandas.nlargest sin parámetro 'key'
- Compatibilidad con versiones antiguas de pandas

---

## [2.6.0] - 2025-11-15

### 🚀 Agregado
- **FASE 11**: Ensemble Model (Voting Classifier)
- Combinación de mejores 3 modelos (LR, RF, XGB)
- Soft voting para promedio de probabilidades
- Evaluación completa con Yellowbrick

---

## [2.5.0] - 2025-11-15

### 🚀 Agregado
- **FASE 10**: Neural Network con TensorFlow/Keras
- Arquitectura optimizada para datos tabulares
- Early stopping y learning rate reduction
- Visualización de curvas de entrenamiento

---

## [2.4.0] - 2025-11-15

### 🚀 Agregado
- **FASE 9**: XGBoost con Optuna
- 20 trials de optimización
- Visualización de optimization history
- Feature importance de XGBoost

---

## [2.3.0] - 2025-11-15

### 🚀 Agregado
- **FASE 8**: Random Forest con Optuna
- Optimización automática de hiperparámetros
- 20 trials con early pruning
- Feature importance analysis

---

## [2.2.0] - 2025-11-14

### 🚀 Agregado
- **FASE 5.7**: Análisis clínico específico
- Análisis detallado por grupos de edad
- Perfiles de riesgo cardiovascular
- Correlaciones clínicas relevantes

---

## [2.1.0] - 2025-11-14

### 🚀 Agregado
- **FASE 7**: Logistic Regression (Baseline Model)
- Modelo baseline simple e interpretable
- Evaluación completa con métricas estándar
- Visualizaciones con Yellowbrick
- Registro automático en MLflow

---

## [2.0.0] - 2025-11-14

### 🚀 Agregado
- **FASE 6**: Preprocesamiento de Datos Profesional
- Train-test split estratificado (80/20)
- Estandarización con StandardScaler
- SMOTE para balanceo de clases
- Prevención de data leakage

### ⚡ Mejorado
- Documentación exhaustiva del preprocesamiento
- Código modular y reutilizable

---

## [1.9.0] - 2025-11-14

### 🚀 Agregado
- **FASE 5.9**: Visualizaciones Multivariadas
- Implementación de PCA (2D y 3D)
- t-SNE para reducción dimensional
- UMAP para visualización
- Gráficos interactivos con Plotly

---

## [1.8.0] - 2025-11-14

### 🚀 Agregado
- **FASE 5.8**: Matriz de Correlación Mejorada
- Heatmap con seaborn
- Identificación de correlaciones fuertes
- Análisis de multicolinealidad

---

## [1.7.1] - 2025-11-14

### 🔧 Corregido
- **FASE 5.7.1**: Corrección de análisis por grupos de edad
- Ajuste de rangos etarios
- Mejora en visualizaciones

---

## [1.7.0] - 2025-11-14

### 🚀 Agregado
- **FASE 5.7.1**: Análisis detallado por grupos de edad
- 4 grupos etarios: <45, 45-54, 55-64, 65+
- Análisis estadístico por grupo
- Visualizaciones comparativas

---

## [1.6.0] - 2025-11-13

### 🚀 Agregado
- **FASE 5.6**: Análisis Bivariado (Features vs Target)
- Visualizaciones comparativas por clase
- Box plots y violin plots
- Análisis estadístico de diferencias

---

## [1.5.0] - 2025-11-13

### 🚀 Agregado
- **FASE 5.5**: Detección y Análisis de Outliers
- Método IQR para detección
- Visualización con box plots
- Análisis de impacto de outliers

---

## [1.4.0] - 2025-11-13

### 🚀 Agregado
- **FASE 5.4**: Ranking de Features con Yellowbrick
- Feature importances visualization
- Identificación de features más relevantes

---

## [1.3.0] - 2025-11-13

### 🚀 Agregado
- **FASE 5.3**: Análisis Univariado - Distribuciones
- Histogramas para variables numéricas
- Análisis de normalidad
- Identificación de patrones

---

## [1.2.0] - 2025-11-13

### 🚀 Agregado
- **FASE 5.2**: Distribución de la Variable Target
- Análisis de balance de clases
- Visualizaciones de distribución

---

## [1.1.0] - 2025-11-13

### 🚀 Agregado
- **FASE 5.1**: Información General y Calidad de Datos
- Análisis de valores faltantes
- Estadísticas descriptivas
- Tipos de datos

---

## [1.0.0] - 2025-11-13

### 🚀 Agregado
- **FASE 1**: Instalación de Dependencias
  - Setup completo de librerías MLOps
  - Optuna, MLflow, Yellowbrick, SHAP, Evidently, Deepchecks
  
- **FASE 2**: Imports y Configuración
  - Configuración de seeds para reproducibilidad
  - Imports organizados por categoría
  
- **FASE 3**: MLflow Configuración con ngrok
  - Setup de MLflow UI
  - Integración con ngrok para acceso remoto
  - Sistema de tracking de experimentos
  
- **FASE 4**: Carga y Exploración de Datos
  - Carga del dataset UCI Heart Disease
  - Conversión a clasificación binaria
  - Vista previa inicial

---

## [0.1.0] - 2025-11-12

### 🚀 Inicial
- Creación del proyecto
- Estructura base del notebook
- Documentación inicial

---

## Tipos de Cambios

- **🚀 Agregado** - Para nuevas funcionalidades
- **⚡ Mejorado** - Para cambios en funcionalidades existentes
- **🔧 Corregido** - Para corrección de bugs
- **🗑️ Eliminado** - Para funcionalidades removidas
- **🔒 Seguridad** - Para vulnerabilidades corregidas
- **📚 Documentación** - Para cambios en documentación

---

## Versionado Semántico

El proyecto sigue [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Cambios incompatibles con versiones anteriores
- **MINOR** (0.X.0): Nuevas funcionalidades retrocompatibles
- **PATCH** (0.0.X): Correcciones de bugs retrocompatibles

---

## Próximas Versiones Planificadas

### [3.5.0] - Planificado
- [ ] Análisis de calibración de modelos
- [ ] Cross-validation estratificada (5-fold)
- [ ] Regularización mejorada en XGBoost
- [ ] Optimización de arquitectura Neural Network

### [4.0.0] - Planificado
- [ ] API REST con FastAPI
- [ ] Containerización con Docker
- [ ] CI/CD con GitHub Actions
- [ ] Deployment en cloud (AWS/GCP/Azure)
- [ ] Sistema de monitoreo en producción

### [4.1.0] - Planificado
- [ ] Soporte para LightGBM y CatBoost
- [ ] AutoML con H2O o Auto-sklearn
- [ ] Feature engineering automatizado
- [ ] Sistema de alertas de drift

---

## Mantenedores

- **Eduardo Farías Reyes** - *Autor Principal* - [GitHub](https://github.com/efarias)

---

## Contribuidores

¿Quieres contribuir? Ver [CONTRIBUTING.md](CONTRIBUTING.md) para lineamientos.

---

**Última actualización:** 17 de Noviembre de 2025
