# 🎯 Interfaz Gráfica de Optimización No Lineal

## Descripción

Esta es la interfaz gráfica moderna y minimalista para el programa de optimización no lineal. Utiliza **CustomTkinter** para proporcionar una experiencia de usuario estética y funcional.

## 🚀 Características de la GUI

### ✨ Diseño Moderno
- **Tema oscuro** por defecto para reducir la fatiga visual
- **Interfaz minimalista** con elementos bien organizados
- **Tipografía clara** y legible
- **Colores consistentes** siguiendo las mejores prácticas de UX

### 📊 Funcionalidades

#### Panel de Configuración (Izquierdo)
1. **Selección de Método**:
   - 🔢 Sin Restricciones (Cálculo Diferencial)
   - 🔗 Con Restricciones de Igualdad (Multiplicadores de Lagrange)
   - ⚖️ Con Restricciones de Desigualdad (Condiciones KKT)

2. **Entrada de Datos**:
   - Campo para variables (ej: x, y, z)
   - Campo para función objetivo
   - Campos dinámicos para restricciones según el método seleccionado

3. **Gestión de Restricciones**:
   - Botones para agregar restricciones
   - Lista visual de restricciones agregadas
   - Diferenciación entre restricciones de igualdad y desigualdad

4. **Controles de Acción**:
   - 🚀 **Resolver**: Ejecuta la optimización
   - 📋 **Cargar Ejemplo**: Carga ejemplos predefinidos
   - 🗑️ **Limpiar**: Limpia todos los campos

#### Panel de Resultados (Derecho)
1. **Área de Resultados**:
   - Texto con scroll para mostrar resultados detallados
   - Formato de consola para mejor legibilidad
   - Colores que contrastan bien con el tema oscuro

2. **Barra de Progreso**:
   - Indicador visual del progreso de la optimización
   - Feedback inmediato al usuario

### 🔧 Características Técnicas

#### Arquitectura
- **Programación orientada a objetos** con clase principal `OptimizationGUI`
- **Separación de responsabilidades** entre UI y lógica de optimización
- **Threading** para evitar bloqueo de la interfaz durante cálculos
- **Manejo de errores** robusto con mensajes informativos

#### Integración
- **Importación directa** de los módulos de optimización existentes
- **Captura de salida** de los algoritmos para mostrar en la GUI
- **Validación de entrada** antes de ejecutar optimizaciones
- **Ejemplos predefinidos** para cada método

## 📋 Uso de la Interfaz

### Paso 1: Seleccionar Método
1. Elige el método de optimización apropiado:
   - **Sin restricciones**: Para problemas de optimización libre
   - **Lagrange**: Para problemas con restricciones de igualdad
   - **KKT**: Para problemas con restricciones de desigualdad

### Paso 2: Configurar el Problema
1. **Variables**: Ingresa las variables separadas por comas (ej: `x, y`)
2. **Función Objetivo**: Define la función a optimizar (ej: `x**2 + y**2`)
3. **Restricciones**: Agrega las restricciones según el método seleccionado

### Paso 3: Resolver
1. Haz clic en **🚀 Resolver**
2. Observa la barra de progreso
3. Revisa los resultados en el panel derecho

### Ejemplos Rápidos
Usa el botón **📋 Cargar Ejemplo** para cargar automáticamente:
- Problemas típicos para cada método
- Configuraciones válidas
- Casos de prueba funcionales

## 🛠️ Instalación y Ejecución

### Requisitos
```bash
pip install customtkinter>=5.2.0
```

### Ejecución
```bash
python gui_app.py
```

## 🎨 Personalización

### Temas Disponibles
- **Modo de Apariencia**: `"dark"`, `"light"`, `"system"`
- **Tema de Color**: `"blue"`, `"green"`, `"dark-blue"`

### Modificar Tema
En el archivo `gui_app.py`, líneas 23-24:
```python
ctk.set_appearance_mode("dark")  # Cambiar aquí
ctk.set_default_color_theme("blue")  # Cambiar aquí
```

## 🔍 Características Avanzadas

### Validación de Entrada
- Verificación de formato de variables
- Validación de restricciones según el método
- Mensajes de error informativos

### Manejo de Errores
- Captura de excepciones durante la optimización
- Mensajes de error claros y específicos
- Recuperación graceful de errores

### Rendimiento
- **Threading**: Los cálculos se ejecutan en hilos separados
- **Responsive UI**: La interfaz permanece responsiva durante cálculos
- **Feedback visual**: Barra de progreso y actualizaciones en tiempo real

## 📱 Compatibilidad

- **Windows**: ✅ Totalmente compatible
- **macOS**: ✅ Compatible
- **Linux**: ✅ Compatible

## 🚀 Ventajas de la GUI

1. **Facilidad de Uso**: No requiere conocimiento de línea de comandos
2. **Visualización Clara**: Resultados organizados y fáciles de leer
3. **Eficiencia**: Carga rápida de ejemplos y validación automática
4. **Accesibilidad**: Interfaz intuitiva para usuarios de todos los niveles
5. **Productividad**: Permite experimentar rápidamente con diferentes configuraciones

## 🔮 Futuras Mejoras

- **Gráficos**: Visualización de funciones y puntos óptimos
- **Exportación**: Guardar resultados en archivos
- **Historial**: Mantener registro de optimizaciones anteriores
- **Plantillas**: Guardar y cargar configuraciones personalizadas
- **Ayuda Contextual**: Tooltips y ayuda integrada

---

**Nota**: Esta interfaz gráfica mantiene toda la funcionalidad del programa de consola original, pero la presenta de manera más accesible y visualmente atractiva.