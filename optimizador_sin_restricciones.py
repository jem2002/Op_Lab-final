import sympy as sp
from sympy import symbols, diff, solve, pprint, latex
import numpy as np
from typing import List, Dict, Tuple

class OptimizadorNoLineal:
    """
    Clase para resolver problemas de optimización no lineal sin restricciones
    """
    
    def __init__(self):
        self.variables = []
        self.funcion_objetivo = None
        self.gradiente = []
        self.hessiana = None
        self.puntos_criticos = []
        self.clasificacion_puntos = []
    
    def definir_variables(self, nombres_variables: List[str]):
        """
        Define las variables simbólicas para la optimización
        
        Args:
            nombres_variables: Lista de nombres de variables (ej: ['x', 'y', 'z'])
        """
        self.variables = symbols(nombres_variables)
        if len(nombres_variables) == 1:
            self.variables = [self.variables]  # Convertir a lista si es una sola variable
        print(f"Variables definidas: {self.variables}")
        return self.variables
    
    def definir_funcion_objetivo(self, expresion_str: str):
        """
        Define la función objetivo a partir de una expresión string
        
        Args:
            expresion_str: Expresión matemática como string (ej: 'x**2 + y**2 - 2*x*y')
        """
        try:
            # Crear un diccionario con las variables para evaluar la expresión
            variables_dict = {str(var): var for var in self.variables}
            
            # Parsear la expresión
            self.funcion_objetivo = sp.sympify(expresion_str, locals=variables_dict)
            
            print("\nFunción objetivo definida:")
            print(f"f({', '.join(str(v) for v in self.variables)}) = {self.funcion_objetivo}")
            
            return self.funcion_objetivo
            
        except Exception as e:
            print(f"Error al definir la función objetivo: {e}")
            return None
    
    def calcular_gradiente(self):
        """
        Calcula el gradiente de la función objetivo
        
        Returns:
            Lista con las derivadas parciales
        """
        if self.funcion_objetivo is None:
            print("Error: Primero debe definir una función objetivo")
            return None
        
        self.gradiente = []
        
        print("\nCalculando gradiente...")
        print("∇f = [")
        
        for i, var in enumerate(self.variables):
            derivada = diff(self.funcion_objetivo, var)
            self.gradiente.append(derivada)
            print(f"  ∂f/∂{var} = {derivada}")
        
        print("]")
        
        return self.gradiente
    
    def encontrar_puntos_criticos(self):
        """
        Encuentra los puntos críticos resolviendo ∇f = 0
        
        Returns:
            Lista de puntos críticos
        """
        if not self.gradiente:
            print("Error: Primero debe calcular el gradiente")
            return None
        
        print("\nResolviendo el sistema ∇f = 0...")
        
        try:
            # Resolver el sistema de ecuaciones ∇f = 0
            soluciones = solve(self.gradiente, self.variables)
            
            if isinstance(soluciones, dict):
                # Una sola solución
                self.puntos_criticos = [soluciones]
            elif isinstance(soluciones, list):
                # Múltiples soluciones
                self.puntos_criticos = soluciones
            else:
                self.puntos_criticos = []
            
            return self.puntos_criticos
            
        except Exception as e:
            print(f"Error al resolver el sistema: {e}")
            return None
    
    def calcular_hessiana(self):
        """
        Calcula la matriz Hessiana (matriz de segundas derivadas parciales)
        
        Returns:
            Matriz Hessiana como matriz de SymPy
        """
        if self.funcion_objetivo is None:
            print("Error: Primero debe definir una función objetivo")
            return None
        
        n = len(self.variables)
        self.hessiana = sp.zeros(n, n)
        
        print("\nCalculando matriz Hessiana...")
        print("H = [")
        
        for i in range(n):
            for j in range(n):
                # Calcular la segunda derivada parcial ∂²f/∂xi∂xj
                segunda_derivada = diff(self.funcion_objetivo, self.variables[i], self.variables[j])
                self.hessiana[i, j] = segunda_derivada
                
        # Mostrar la matriz Hessiana
        for i in range(n):
            fila = "  ["
            for j in range(n):
                if j > 0:
                    fila += ", "
                fila += f"∂²f/∂{self.variables[i]}∂{self.variables[j]} = {self.hessiana[i, j]}"
            fila += "]"
            print(fila)
        
        print("]")
        
        return self.hessiana
    
    def clasificar_punto_critico(self, punto):
        """
        Clasifica un punto crítico usando el criterio de la segunda derivada
        
        Args:
            punto: Diccionario o lista con los valores del punto crítico
        
        Returns:
            String con la clasificación del punto
        """
        if self.hessiana is None:
            print("Error: Primero debe calcular la matriz Hessiana")
            return "No clasificado"
        
        # Convertir punto a diccionario si es necesario
        if isinstance(punto, (list, tuple)):
            punto_dict = dict(zip(self.variables, punto))
        else:
            punto_dict = punto
        
        # Evaluar la Hessiana en el punto crítico
        hessiana_evaluada = self.hessiana.subs(punto_dict)
        
        n = len(self.variables)
        
        if n == 1:
            # Caso unidimensional: solo verificar el signo de la segunda derivada
            segunda_derivada = float(hessiana_evaluada[0, 0])
            if segunda_derivada > 0:
                return "Mínimo local"
            elif segunda_derivada < 0:
                return "Máximo local"
            else:
                return "Criterio no concluyente (segunda derivada = 0)"
        
        elif n == 2:
            # Caso bidimensional: usar determinante y traza
            try:
                # Convertir a float para evaluación numérica
                h11 = float(hessiana_evaluada[0, 0])
                h12 = float(hessiana_evaluada[0, 1])
                h21 = float(hessiana_evaluada[1, 0])
                h22 = float(hessiana_evaluada[1, 1])
                
                determinante = h11 * h22 - h12 * h21
                traza = h11 + h22
                
                if determinante > 0:
                    if traza > 0:
                        return f"Mínimo local (det={determinante:.4f} > 0, tr={traza:.4f} > 0)"
                    else:
                        return f"Máximo local (det={determinante:.4f} > 0, tr={traza:.4f} < 0)"
                elif determinante < 0:
                    return f"Punto de silla (det={determinante:.4f} < 0)"
                else:
                    return f"Criterio no concluyente (det={determinante:.4f} = 0)"
            
            except (ValueError, TypeError):
                return "No se pudo evaluar numéricamente"
        
        else:
            # Caso multidimensional: verificar definitud usando autovalores
            try:
                autovalores = hessiana_evaluada.eigenvals()
                autovalores_numericos = [complex(val).real for val in autovalores.keys()]
                
                todos_positivos = all(val > 1e-10 for val in autovalores_numericos)
                todos_negativos = all(val < -1e-10 for val in autovalores_numericos)
                
                if todos_positivos:
                    return f"Mínimo local (todos los autovalores > 0: {autovalores_numericos})"
                elif todos_negativos:
                    return f"Máximo local (todos los autovalores < 0: {autovalores_numericos})"
                else:
                    return f"Punto de silla (autovalores mixtos: {autovalores_numericos})"
            
            except Exception as e:
                return f"Error al calcular autovalores: {e}"
    
    def analizar_puntos_criticos(self):
        """
        Analiza y clasifica todos los puntos críticos encontrados
        """
        if not self.puntos_criticos:
            print("Error: Primero debe encontrar los puntos críticos")
            return
        
        if self.hessiana is None:
            self.calcular_hessiana()
        
        self.clasificacion_puntos = []
        
        print("\nClasificando puntos críticos usando el criterio de la segunda derivada...")
        
        for i, punto in enumerate(self.puntos_criticos):
            clasificacion = self.clasificar_punto_critico(punto)
            self.clasificacion_puntos.append(clasificacion)
            print(f"\nPunto crítico {i+1}: {clasificacion}")
    
    def mostrar_puntos_criticos(self):
        """
        Muestra los puntos críticos encontrados con su clasificación
        """
        if not self.puntos_criticos:
            print("\nNo se encontraron puntos críticos o no se han calculado aún.")
            return
        
        print("\n" + "="*70)
        print("PUNTOS CRÍTICOS ENCONTRADOS Y CLASIFICADOS")
        print("="*70)
        
        for i, punto in enumerate(self.puntos_criticos, 1):
            print(f"\nPunto crítico {i}:")
            
            if isinstance(punto, dict):
                for var, valor in punto.items():
                    print(f"  {var} = {valor}")
                
                # Evaluar la función en este punto
                valor_funcion = self.funcion_objetivo.subs(punto)
                print(f"  f({', '.join(str(v) for v in punto.values())}) = {valor_funcion}")
            
            elif isinstance(punto, (list, tuple)):
                for j, valor in enumerate(punto):
                    print(f"  {self.variables[j]} = {valor}")
                
                # Crear diccionario para evaluación
                punto_dict = dict(zip(self.variables, punto))
                valor_funcion = self.funcion_objetivo.subs(punto_dict)
                print(f"  f({', '.join(str(v) for v in punto)}) = {valor_funcion}")
            
            # Mostrar clasificación si está disponible
            if i-1 < len(self.clasificacion_puntos):
                print(f"  Clasificación: {self.clasificacion_puntos[i-1]}")
            else:
                print("  Clasificación: No calculada")
    
    def analisis_completo(self, nombres_variables: List[str], expresion_funcion: str):
        """
        Realiza el análisis completo de optimización
        
        Args:
            nombres_variables: Lista de nombres de variables
            expresion_funcion: Expresión de la función objetivo
        """
        print("\n" + "="*60)
        print("ANÁLISIS DE OPTIMIZACIÓN NO LINEAL SIN RESTRICCIONES")
        print("="*60)
        
        # Paso 1: Definir variables
        self.definir_variables(nombres_variables)
        
        # Paso 2: Definir función objetivo
        if self.definir_funcion_objetivo(expresion_funcion) is None:
            return
        
        # Paso 3: Calcular gradiente
        if self.calcular_gradiente() is None:
            return
        
        # Paso 4: Encontrar puntos críticos
        if self.encontrar_puntos_criticos() is None:
            return
        
        # Paso 5: Calcular matriz Hessiana
        if self.calcular_hessiana() is None:
            return
        
        # Paso 6: Analizar y clasificar puntos críticos
        self.analizar_puntos_criticos()
        
        # Paso 7: Mostrar resultados
        self.mostrar_puntos_criticos()

def mostrar_ejemplos_sin_restricciones(optimizador):
    """
    Muestra ejemplos predefinidos sin restricciones
    """
    ejemplos = [
        {
            "nombre": "Función cuadrática simple",
            "variables": ["x", "y"],
            "funcion": "x**2 + y**2",
            "descripcion": "Mínimo global en el origen"
        },
        {
            "nombre": "Función de Rosenbrock (2D)",
            "variables": ["x", "y"],
            "funcion": "(1-x)**2 + 100*(y-x**2)**2",
            "descripcion": "Función clásica de optimización"
        },
        {
            "nombre": "Función con punto de silla",
            "variables": ["x", "y"],
            "funcion": "x**2 - y**2",
            "descripcion": "Punto de silla en el origen"
        },
        {
            "nombre": "Función cúbica",
            "variables": ["x", "y"],
            "funcion": "x**3 + y**3 - 3*x*y",
            "descripcion": "Múltiples puntos críticos"
        }
    ]
    
    print("\nEjemplos SIN restricciones disponibles:")
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"{i}. {ejemplo['nombre']}")
        print(f"   Variables: {ejemplo['variables']}")
        print(f"   Función: {ejemplo['funcion']}")
        print(f"   Descripción: {ejemplo['descripcion']}\n")
    
    try:
        seleccion = int(input("Seleccione un ejemplo (1-4): ")) - 1
        
        if 0 <= seleccion < len(ejemplos):
            ejemplo = ejemplos[seleccion]
            print(f"\nEjecutando: {ejemplo['nombre']}")
            optimizador.analisis_completo(ejemplo['variables'], ejemplo['funcion'])
        else:
            print("Selección no válida.")
            
    except ValueError:
        print("Por favor, ingrese un número válido.")

def analisis_completo_interactivo(optimizador):
    """
    Realiza un análisis completo de forma interactiva
    """
    try:
        # Solicitar variables
        print("\nIngrese las variables separadas por comas (ej: x,y,z):")
        variables_input = input("Variables: ").strip()
        nombres_variables = [var.strip() for var in variables_input.split(',')]
        
        # Solicitar función objetivo
        print("\nIngrese la función objetivo usando las variables definidas.")
        print("Ejemplos:")
        print("  - x**2 + y**2")
        print("  - x**2 + y**2 - 2*x*y + 3*x")
        print("  - x**3 + y**3 - 3*x*y")
        
        expresion = input("\nFunción objetivo: ").strip()
        
        # Realizar análisis
        optimizador.analisis_completo(nombres_variables, expresion)
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")