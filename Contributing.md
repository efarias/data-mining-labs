# Guía de Contribución

¡Gracias por tu interés en contribuir al Lab Integrador U3! 🎉

Este documento proporciona lineamientos para contribuir al proyecto de manera efectiva.

---

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Lineamientos de Desarrollo](#lineamientos-de-desarrollo)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Estándares de Código](#estándares-de-código)
- [Estructura de Commits](#estructura-de-commits)
- [Reportar Bugs](#reportar-bugs)
- [Sugerir Mejoras](#sugerir-mejoras)

---

## 📜 Código de Conducta

Este proyecto adhiere a un Código de Conducta que todos los participantes deben seguir. Al participar, se espera que mantengas este código.

### Nuestro Compromiso

- Crear un ambiente acogedor e inclusivo
- Respetar diferentes puntos de vista y experiencias
- Aceptar críticas constructivas de manera positiva
- Enfocarnos en lo mejor para la comunidad

---

## 🤝 ¿Cómo Puedo Contribuir?

### 1. Reportar Bugs

Si encuentras un bug, por favor:

1. **Verifica** que no haya sido reportado previamente en [Issues](https://github.com/tuusuario/lab-integrador-u3/issues)
2. **Abre un nuevo issue** usando la plantilla de bug report
3. **Incluye**:
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Screenshots si aplica
   - Versiones de Python y librerías
   - Sistema operativo

### 2. Sugerir Mejoras

Para sugerir una nueva funcionalidad:

1. **Verifica** que no exista una sugerencia similar
2. **Abre un issue** con la etiqueta `enhancement`
3. **Describe**:
   - La funcionalidad propuesta
   - Por qué sería útil
   - Ejemplos de uso
   - Posible implementación

### 3. Mejorar Documentación

La documentación siempre puede mejorar:

- Corregir errores tipográficos
- Clarificar explicaciones
- Agregar ejemplos
- Traducir contenido
- Mejorar comentarios en código

### 4. Contribuir Código

Ver [Proceso de Pull Request](#proceso-de-pull-request)

---

## 🛠️ Lineamientos de Desarrollo

### Configuración del Entorno

```bash
# 1. Fork el repositorio

# 2. Clonar tu fork
git clone https://github.com/tu-usuario/lab-integrador-u3.git
cd lab-integrador-u3

# 3. Agregar upstream
git remote add upstream https://github.com/usuario-original/lab-integrador-u3.git

# 4. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate      # Windows

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Instalar dependencias de desarrollo (opcional)
pip install pytest black flake8 mypy
```

### Estructura de Branches

- `main` - Rama principal (protegida)
- `develop` - Rama de desarrollo
- `feature/*` - Nuevas funcionalidades
- `bugfix/*` - Corrección de bugs
- `hotfix/*` - Correcciones urgentes
- `docs/*` - Cambios en documentación

### Nomenclatura de Branches

```bash
# Ejemplos:
feature/add-lightgbm-model
feature/calibration-analysis
bugfix/fix-shap-visualization
docs/update-readme
hotfix/critical-memory-leak
```

---

## 🔄 Proceso de Pull Request

### Antes de Crear un PR

1. **Sincronizar con upstream:**
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

2. **Crear branch desde develop:**
```bash
git checkout develop
git checkout -b feature/tu-feature
```

3. **Hacer cambios y commits:**
```bash
git add .
git commit -m "feat: descripción del cambio"
```

4. **Mantener branch actualizado:**
```bash
git fetch upstream
git rebase upstream/develop
```

### Crear el Pull Request

1. **Push a tu fork:**
```bash
git push origin feature/tu-feature
```

2. **Abrir PR en GitHub:**
   - Base: `develop` (no `main`)
   - Compare: `feature/tu-feature`
   - Título descriptivo
   - Descripción completa usando la plantilla

3. **Checklist del PR:**
   - [ ] Código sigue los estándares del proyecto
   - [ ] Tests pasan (si aplica)
   - [ ] Documentación actualizada
   - [ ] CHANGELOG.md actualizado
   - [ ] Sin conflictos con develop
   - [ ] PR revisado por ti mismo primero

### Review Process

1. Mantenedores revisarán el PR
2. Pueden solicitar cambios
3. Una vez aprobado, será mergeado
4. Branch será eliminado automáticamente

---

## 📝 Estándares de Código

### Python Style Guide

Seguimos [PEP 8](https://www.python.org/dev/peps/pep-0008/):

```python
# ✅ CORRECTO
def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        
    Returns:
        dict: Diccionario con métricas
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

# ❌ INCORRECTO
def CalculateMetrics(Y_TRUE,Y_PRED):
    ACC=accuracy_score(Y_TRUE,Y_PRED)
    f1=f1_score(Y_TRUE,Y_PRED)
    return {'accuracy':ACC,'f1_score':f1}
```

### Convenciones del Proyecto

1. **Variables:**
   ```python
   # Descriptivas en snake_case
   train_accuracy = 0.95
   best_model_name = "Random Forest"
   ```

2. **Funciones:**
   ```python
   # Verbos en snake_case, con docstrings
   def train_model(X_train, y_train):
       """Entrena un modelo."""
       pass
   ```

3. **Clases:**
   ```python
   # PascalCase con docstrings
   class ModelTrainer:
       """Clase para entrenar modelos."""
       pass
   ```

4. **Constantes:**
   ```python
   # UPPER_SNAKE_CASE
   RANDOM_STATE = 42
   MAX_ITERATIONS = 1000
   ```

5. **Comentarios:**
   ```python
   # Comentarios claros y concisos
   # Explicar el "por qué", no el "qué"
   
   # Usar secciones para organizar código largo
   # ========================================
   # SECCIÓN: Preprocesamiento de Datos
   # ========================================
   ```

### Jupyter Notebooks

```python
# Usar markdown para secciones principales
# Mantener celdas cortas y enfocadas
# Incluir outputs (pero limpiar antes de commit)
# Numerar fases claramente
# Agregar @title para ocultar código en Colab
```

---

## 📦 Estructura de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>(<scope>): <descripción corta>

[cuerpo opcional]

[footer opcional]
```

### Tipos

- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formato (no afecta código)
- `refactor`: Refactorización
- `test`: Agregar o corregir tests
- `chore`: Mantenimiento

### Ejemplos

```bash
# Feat
git commit -m "feat(models): add LightGBM model with Optuna"

# Fix
git commit -m "fix(shap): correct variable name in SHAP analysis"

# Docs
git commit -m "docs(readme): update installation instructions"

# Refactor
git commit -m "refactor(preprocessing): extract scaling to separate function"

# Múltiples líneas
git commit -m "feat(validation): add calibration analysis

- Implement calibration_curve
- Add visualization
- Update documentation"
```

---

## 🐛 Reportar Bugs

### Plantilla de Bug Report

```markdown
**Descripción del Bug**
Descripción clara y concisa del bug.

**Para Reproducir**
Pasos para reproducir:
1. Ir a '...'
2. Ejecutar '...'
3. Ver error

**Comportamiento Esperado**
Qué esperabas que sucediera.

**Screenshots**
Si aplica, agregar screenshots.

**Entorno:**
 - OS: [e.g., Ubuntu 22.04, Windows 11]
 - Python: [e.g., 3.9.7]
 - Librerías: [e.g., scikit-learn 1.2.0]

**Contexto Adicional**
Cualquier otra información relevante.
```

---

## 💡 Sugerir Mejoras

### Plantilla de Feature Request

```markdown
**¿Tu sugerencia está relacionada con un problema?**
Descripción clara del problema.

**Describe la solución que te gustaría**
Descripción clara de lo que quieres que suceda.

**Describe alternativas que hayas considerado**
Otras soluciones o features consideradas.

**Contexto adicional**
Cualquier otra información o screenshots.
```

---

## 🧪 Tests

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_preprocessing.py

# Con coverage
pytest --cov=src tests/
```

### Escribir Tests

```python
import pytest
from src.models import train_model

def test_train_model():
    """Test que el modelo entrena correctamente."""
    X_train = [[1, 2], [3, 4]]
    y_train = [0, 1]
    
    model = train_model(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')
```

---

## 📚 Recursos Adicionales

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Python PEP 8](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## 🙏 Agradecimientos

Agradecemos a todos los contribuidores que ayudan a mejorar este proyecto. Tu tiempo y esfuerzo son valiosos.

### Contribuidores Destacados

<!-- Se llenará automáticamente con contribuidores -->

---

## 📞 Contacto

¿Preguntas sobre cómo contribuir?

- **Issues:** [GitHub Issues](https://github.com/efarias/data-mining-labs/issues)
- **Discusiones:** [GitHub Discussions](https://github.com/efarias/data-mining-labs/discussions)
- **Email:** edufarias@gmail.com

---

**¡Esperamos tus contribuciones!** 🚀

---

**Última actualización:** 17 de Noviembre de 2025
