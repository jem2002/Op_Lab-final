# Optimización No Lineal - Sistema Completo

Este proyecto implementa un sistema completo para resolver problemas de optimización no lineal usando Python y SymPy. El programa maneja los tres casos principales de optimización no lineal:

1. **Sin restricciones** (Cálculo Diferencial)
2. **Con restricciones de igualdad** (Multiplicadores de Lagrange)
3. **Con restricciones de desigualdad** (Condiciones KKT)

## Características

- **Arquitectura modular**: Código organizado en módulos especializados
- **Optimización sin restricciones**: Encuentra puntos críticos usando cálculo diferencial
- **Optimización con restricciones de igualdad**: Implementa el método de multiplicadores de Lagrange
- **Optimización con restricciones de desigualdad**: Implementa las condiciones de Karush-Kuhn-Tucker (KKT)
- **Interfaz interactiva**: Permite al usuario definir funciones y restricciones de forma dinámica
- **Ejemplos predefinidos**: Incluye casos de estudio clásicos para cada método
- **Análisis simbólico**: Utiliza SymPy para cálculos exactos

### Optimización Sin Restricciones
- ✅ Definición de funciones objetivo de múltiples variables de forma simbólica
- ✅ Cálculo automático del gradiente (vector de primeras derivadas parciales)
- ✅ Búsqueda de puntos críticos resolviendo ∇f = 0
- ✅ Visualización clara de los resultados

### Optimización Con Restricciones (Multiplicadores de Lagrange)
- ✅ Definición de función objetivo f(x,y,...) y restricciones g(x,y,...)=0
- ✅ Creación automática de variables de Lagrange (λ)
- ✅ Construcción simbólica de la función Lagrangiana L(x,λ) = f(x) - λ·g(x)
- ✅ Cálculo del gradiente de la Lagrangiana ∇L
- ✅ Resolución del sistema ∇L = 0
- ✅ Mostrar puntos óptimos y valores de multiplicadores

### Características Generales
- ✅ Interfaz interactiva con menú de opciones
- ✅ Ejemplos predefinidos para ambos tipos de optimización
- ✅ Manejo robusto de errores y validación de entrada

## Instalación

1. Clona o descarga este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

O instala manualmente:

```bash
pip install sympy numpy
```

## Uso

### Ejecución del programa

```bash
python app.py
```

### Opciones del menú

1. **Optimización SIN restricciones**: Análisis clásico de puntos críticos
2. **Optimización CON restricciones**: Método de multiplicadores de Lagrange
3. **Ejemplos predefinidos**: Casos de estudio para ambos tipos
4. **Salir**: Terminar el programa

## Ejemplos de uso

### Ejemplos Sin Restricciones

#### Ejemplo 1: Función cuadrática simple
**Variables**: x, y  
**Función**: `x**2 + y**2`  
**Resultado**: Mínimo global en (0, 0)

#### Ejemplo 2: Función de Rosenbrock
**Variables**: x, y  
**Función**: `(1-x)**2 + 100*(y-x**2)**2`  
**Resultado**: Mínimo global en (1, 1)

#### Ejemplo 3: Función con punto de silla
**Variables**: x, y  
**Función**: `x**2 - y**2`  
**Resultado**: Punto de silla en (0, 0)

### Ejemplos Con Restricciones (Multiplicadores de Lagrange)

#### Ejemplo 1: Optimización en un círculo
**Variables**: x, y  
**Función objetivo**: `x + y`  
**Restricción**: `x**2 + y**2 - 1 = 0`  
**Problema**: Maximizar x+y sujeto a x²+y²=1

#### Ejemplo 2: Mínimo de distancia al origen
**Variables**: x, y  
**Función objetivo**: `x**2 + y**2`  
**Restricción**: `x + y - 2 = 0`  
**Problema**: Minimizar x²+y² sujeto a x+y=2

#### Ejemplo 3: Función de utilidad económica
**Variables**: x, y  
**Función objetivo**: `x*y`  
**Restricción**: `2*x + 3*y - 12 = 0`  
**Problema**: Maximizar xy sujeto a 2x+3y=12 (restricción presupuestaria)

## Sintaxis para funciones

El programa acepta expresiones matemáticas usando la sintaxis de Python/SymPy:

- **Potencias**: `x**2`, `y**3`
- **Multiplicación**: `2*x*y`, `3*x`
- **Funciones**: `sin(x)`, `cos(y)`, `exp(x)`, `log(x)`
- **Constantes**: `pi`, `E`

### Ejemplos de expresiones válidas:

```python
# Función cuadrática
"x**2 + y**2 + 2*x*y"

# Función cúbica
"x**3 + y**3 - 3*x*y"

# Con funciones trigonométricas
"sin(x)**2 + cos(y)**2"

# Con exponenciales
"exp(x**2 + y**2)"

# Función de múltiples variables
"x**2 + y**2 + z**2 - 2*x*y + 3*z"
```

## Metodología

### 1. Optimización Sin Restricciones (Cálculo Diferencial)
1. **Definir el problema**: Variables x y función objetivo f(x)
2. **Calcular gradiente**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
3. **Encontrar puntos críticos**: Resolver ∇f = 0
4. **Analizar resultados**: Evaluar la función en los puntos críticos

### 2. Optimización Con Restricciones de Igualdad (Multiplicadores de Lagrange)
1. **Definir el problema**: Variables x, función objetivo f(x), restricciones g(x) = 0
2. **Crear multiplicadores**: Variables λᵢ para cada restricción
3. **Función Lagrangiana**: L(x,λ) = f(x) - Σ(λᵢ·gᵢ(x))
4. **Gradiente de L**: ∇L con respecto a todas las variables
5. **Sistema de ecuaciones**: ∇L = 0
6. **Condiciones de optimalidad**:
   - ∂L/∂xᵢ = 0 (condiciones de primer orden)
   - ∂L/∂λᵢ = -gᵢ(x) = 0 (restricciones)
7. **Análisis de resultados**: Puntos óptimos y valores de multiplicadores

### 3. Optimización Con Restricciones de Desigualdad (Condiciones KKT)
1. **Definir el problema**: Variables x, función objetivo f(x), restricciones g(x) ≤ 0, h(x) = 0
2. **Crear multiplicadores**: Variables μᵢ ≥ 0 para desigualdades, λⱼ para igualdades
3. **Lagrangiana extendida**: L(x,μ,λ) = f(x) + Σ(μᵢ·gᵢ(x)) + Σ(λⱼ·hⱼ(x))
4. **Condiciones KKT**:
   - ∇ₓL = 0 (estacionariedad)
   - gᵢ(x) ≤ 0 (factibilidad primal)
   - hⱼ(x) = 0 (restricciones de igualdad)
   - μᵢ ≥ 0 (factibilidad dual)
   - μᵢ·gᵢ(x) = 0 (holgura complementaria)
5. **Verificación**: Comprobar que todas las condiciones se cumplen

## Estructura del Código

El proyecto está organizado en cuatro módulos principales:

### `app.py` - Programa Principal
- Menú interactivo unificado
- Integración de todos los métodos de optimización
- Verificación de dependencias
- Manejo de errores y navegación

### `optimizador_sin_restricciones.py`
Maneja la optimización sin restricciones usando cálculo diferencial:
- `OptimizadorNoLineal`: Clase principal
- `definir_variables()`: Define las variables simbólicas
- `definir_funcion_objetivo()`: Establece la función a optimizar
- `calcular_gradiente()`: Calcula ∇f
- `encontrar_puntos_criticos()`: Resuelve ∇f = 0
- `mostrar_puntos_criticos()`: Presenta los resultados

### `optimizador_lagrange.py`
Maneja la optimización con restricciones de igualdad usando multiplicadores de Lagrange:
- `OptimizadorConRestricciones`: Clase principal
- `definir_variables()`: Define las variables simbólicas
- `definir_funcion_objetivo()`: Establece la función objetivo
- `agregar_restriccion()`: Añade restricciones g(x) = 0
- `crear_multiplicadores_lagrange()`: Crea variables λ
- `construir_lagrangiana()`: Construye L(x,λ) = f(x) - Σ(λᵢ·gᵢ(x))
- `calcular_gradiente_lagrangiana()`: Calcula ∇L
- `resolver_sistema_lagrange()`: Resuelve ∇L = 0
- `mostrar_puntos_optimos()`: Presenta los resultados

### `optimizador_kkt.py`
Maneja la optimización con restricciones de desigualdad usando condiciones KKT:
- `OptimizadorKKT`: Clase principal
- `definir_variables()`: Define las variables simbólicas
- `definir_funcion_objetivo()`: Establece la función objetivo
- `agregar_restriccion_igualdad()`: Añade restricciones h(x) = 0
- `agregar_restriccion_desigualdad()`: Añade restricciones g(x) ≤ 0
- `crear_multiplicadores_kkt()`: Crea variables λ y μ
- `construir_lagrangiana_kkt()`: Construye la Lagrangiana extendida
- `generar_condiciones_kkt()`: Genera las condiciones de optimalidad
- `verificar_condiciones_kkt()`: Verifica las condiciones en las soluciones

## Limitaciones

- **Análisis de segundo orden**: No implementa clasificación completa de puntos críticos usando la matriz Hessiana
- **Optimización global**: No garantiza encontrar el óptimo global, solo puntos críticos/estacionarios
- **Restricciones complejas**: Puede tener dificultades con sistemas de restricciones muy complejos o no lineales
- **Condiciones KKT**: La verificación de condiciones KKT es básica y puede requerir análisis manual adicional
- **Soluciones numéricas**: Depende de la capacidad de SymPy para resolver sistemas simbólicamente

## Extensiones Posibles

- **Análisis de segundo orden**: Implementar clasificación completa usando la matriz Hessiana y condiciones de segundo orden
- **Optimización numérica**: Integrar métodos numéricos (scipy.optimize) para casos donde el análisis simbólico falla
- **Visualización**: Agregar gráficos 2D/3D para funciones de 2-3 variables
- **Verificación automática KKT**: Mejorar la verificación automática de condiciones KKT
- **Exportación**: Permitir exportar resultados a LaTeX, PDF o otros formatos
- **Interfaz gráfica**: Desarrollar una GUI para facilitar el uso
- **Optimización estocástica**: Agregar métodos para optimización bajo incertidumbre

## Dependencias

- **SymPy**: Matemática simbólica
- **NumPy**: Computación numérica (opcional, para extensiones futuras)

## Autor

Script desarrollado para resolver problemas de optimización no lineal sin restricciones usando SymPy.

## Licencia

Este proyecto es de uso educativo y está disponible bajo licencia MIT.