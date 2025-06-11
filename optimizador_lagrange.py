import sympy as sp
from sympy import symbols, diff, solve
from typing import List
from clasificador_puntos import ClasificadorPuntos
from formateador_didactico import FormateadorDidactico

class OptimizadorConRestricciones:
    """
    Clase para resolver problemas de optimización no lineal con restricciones de igualdad
    usando el método de los Multiplicadores de Lagrange
    """
    
    def __init__(self):
        self.variables = []
        self.funcion_objetivo = None
        self.restricciones = []
        self.multiplicadores = []
        self.lagrangiana = None
        self.gradiente_lagrangiana = []
        self.hessiana_lagrangiana = None
        self.hessiana_orlada = None
        self.puntos_optimos = []
        self.clasificacion_puntos = []
        self.clasificador = ClasificadorPuntos()
        self.formateador = FormateadorDidactico()
    
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
            expresion_str: Expresión matemática como string
        """
        try:
            variables_dict = {str(var): var for var in self.variables}
            self.funcion_objetivo = sp.sympify(expresion_str, locals=variables_dict)
            
            print("\nFunción objetivo definida:")
            print(f"f({', '.join(str(v) for v in self.variables)}) = {self.funcion_objetivo}")
            
            return self.funcion_objetivo
            
        except Exception as e:
            print(f"Error al definir la función objetivo: {e}")
            return None
    
    def agregar_restriccion(self, expresion_str: str):
        """
        Agrega una restricción de igualdad g(x) = 0
        
        Args:
            expresion_str: Expresión de la restricción (se asume = 0)
        """
        try:
            variables_dict = {str(var): var for var in self.variables}
            restriccion = sp.sympify(expresion_str, locals=variables_dict)
            self.restricciones.append(restriccion)
            
            print(f"\nRestricción {len(self.restricciones)} agregada:")
            print(f"g_{len(self.restricciones)}({', '.join(str(v) for v in self.variables)}) = {restriccion} = 0")
            
            return restriccion
            
        except Exception as e:
            print(f"Error al agregar restricción: {e}")
            return None
    
    def crear_multiplicadores_lagrange(self):
        """
        Crea las variables de multiplicadores de Lagrange (λ)
        """
        num_restricciones = len(self.restricciones)
        if num_restricciones == 0:
            print("Error: No hay restricciones definidas")
            return None
        
        # Crear multiplicadores λ1, λ2, ..., λn
        nombres_lambda = [f'lambda_{i+1}' for i in range(num_restricciones)]
        self.multiplicadores = symbols(nombres_lambda)
        
        # Asegurar que siempre sea una lista
        if not isinstance(self.multiplicadores, (list, tuple)):
            self.multiplicadores = [self.multiplicadores]
        
        print("\nMultiplicadores de Lagrange creados:")
        for i, lam in enumerate(self.multiplicadores):
            print(f"  λ_{i+1} = {lam}")
        
        return self.multiplicadores
    
    def construir_lagrangiana(self):
        """
        Construye la función Lagrangiana: L(x,λ) = f(x) - Σ(λᵢ * gᵢ(x))
        """
        if self.funcion_objetivo is None:
            print("Error: Primero debe definir la función objetivo")
            return None
        
        if not self.restricciones:
            print("Error: Primero debe definir las restricciones")
            return None
        
        if not self.multiplicadores:
            self.crear_multiplicadores_lagrange()
        
        # L(x,λ) = f(x) - Σ(λᵢ * gᵢ(x))
        self.lagrangiana = self.funcion_objetivo
        
        print("\nConstruyendo función Lagrangiana:")
        print(f"L = {self.funcion_objetivo}")
        
        for i, (lam, restriccion) in enumerate(zip(self.multiplicadores, self.restricciones)):
            # Convertir la restricción string a expresión simbólica si es necesario
            if isinstance(restriccion, str):
                restriccion_expr = sp.sympify(restriccion)
            else:
                restriccion_expr = restriccion
            
            termino = lam * restriccion_expr
            self.lagrangiana -= termino
            print(f"    - {lam} * ({restriccion_expr})")
        
        print(f"\nLagrangiana completa:")
        print(f"L({', '.join(str(v) for v in self.variables)}, {', '.join(str(l) for l in self.multiplicadores)}) = {self.lagrangiana}")
        
        return self.lagrangiana
    
    def calcular_gradiente_lagrangiana(self):
        """
        Calcula el gradiente de la Lagrangiana con respecto a todas las variables
        (incluyendo los multiplicadores de Lagrange)
        """
        if self.lagrangiana is None:
            print("Error: Primero debe construir la Lagrangiana")
            return None
        
        self.gradiente_lagrangiana = []
        todas_las_variables = list(self.variables) + list(self.multiplicadores)
        
        print("\nCalculando gradiente de la Lagrangiana:")
        print("∇L = [")
        
        # Derivadas con respecto a las variables originales
        for var in self.variables:
            derivada = diff(self.lagrangiana, var)
            self.gradiente_lagrangiana.append(derivada)
            print(f"  ∂L/∂{var} = {derivada}")
        
        # Derivadas con respecto a los multiplicadores (que son las restricciones)
        for i, lam in enumerate(self.multiplicadores):
            derivada = diff(self.lagrangiana, lam)
            self.gradiente_lagrangiana.append(derivada)
            print(f"  ∂L/∂{lam} = {derivada}")
        
        print("]")
        
        return self.gradiente_lagrangiana
    
    def resolver_sistema_lagrange(self):
        """
        Resuelve el sistema de ecuaciones ∇L = 0
        """
        if not self.gradiente_lagrangiana:
            print("Error: Primero debe calcular el gradiente de la Lagrangiana")
            return None
        
        todas_las_variables = list(self.variables) + list(self.multiplicadores)
        
        print("\nResolviendo el sistema ∇L = 0...")
        print("Sistema de ecuaciones:")
        for i, eq in enumerate(self.gradiente_lagrangiana):
            print(f"  Ecuación {i+1}: {eq} = 0")
        
        try:
            soluciones = solve(self.gradiente_lagrangiana, todas_las_variables)
            
            if isinstance(soluciones, dict):
                self.puntos_optimos = [soluciones]
            elif isinstance(soluciones, list):
                self.puntos_optimos = soluciones
            else:
                self.puntos_optimos = []
            
            return self.puntos_optimos

        except Exception as e:
            print(f"Error al resolver el sistema: {e}")
            return None
    
    def calcular_hessiana_lagrangiana(self):
        """
        Calcula la matriz Hessiana de la Lagrangiana con respecto a las variables originales
        
        Returns:
            Matriz Hessiana de la Lagrangiana
        """
        if self.lagrangiana is None:
            print("Error: Primero debe construir la Lagrangiana")
            return None
        
        n = len(self.variables)
        self.hessiana_lagrangiana = sp.zeros(n, n)
        
        print("\nCalculando matriz Hessiana de la Lagrangiana...")
        print("H_L = [")
        
        for i in range(n):
            for j in range(n):
                # Calcular la segunda derivada parcial ∂²L/∂xi∂xj
                segunda_derivada = diff(self.lagrangiana, self.variables[i], self.variables[j])
                self.hessiana_lagrangiana[i, j] = segunda_derivada
        
        # Mostrar la matriz Hessiana
        for i in range(n):
            fila = "  ["
            for j in range(n):
                if j > 0:
                    fila += ", "
                fila += f"∂²L/∂{self.variables[i]}∂{self.variables[j]} = {self.hessiana_lagrangiana[i, j]}"
            fila += "]"
            print(fila)
        
        print("]")
        
        return self.hessiana_lagrangiana
    
    def calcular_hessiana_orlada(self):
        """
        Calcula la matriz Hessiana orlada (bordered Hessian) para el análisis de segunda derivada
        con restricciones de igualdad
        
        Returns:
            Matriz Hessiana orlada
        """
        if self.hessiana_lagrangiana is None:
            self.calcular_hessiana_lagrangiana()
        
        n = len(self.variables)  # número de variables
        m = len(self.restricciones)  # número de restricciones
        
        # La matriz orlada tiene dimensión (n+m) x (n+m)
        self.hessiana_orlada = sp.zeros(n + m, n + m)
        
        print("\nCalculando matriz Hessiana orlada...")
        
        # Bloque superior izquierdo: Hessiana de la Lagrangiana
        for i in range(n):
            for j in range(n):
                self.hessiana_orlada[i, j] = self.hessiana_lagrangiana[i, j]
        
        # Bloques de restricciones
        for k, restriccion in enumerate(self.restricciones):
            # Bloque superior derecho: gradientes de restricciones
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada[i, n + k] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada[n + k, j] = grad_restriccion
        
        # Bloque inferior derecho: ceros (para restricciones de igualdad)
        for i in range(m):
            for j in range(m):
                self.hessiana_orlada[n + i, n + j] = 0
        
        print(f"Matriz Hessiana orlada de dimensión {n+m}x{n+m}:")
        print("H_orlada = [")
        for i in range(n + m):
            fila = "  ["
            for j in range(n + m):
                if j > 0:
                    fila += ", "
                fila += f"{self.hessiana_orlada[i, j]}"
            fila += "]"
            print(fila)
        print("]")
        
        return self.hessiana_orlada
    
    def clasificar_punto_con_restricciones(self, punto):
        """
        Clasifica un punto crítico con restricciones usando el criterio de la segunda derivada
        (análisis de la matriz Hessiana orlada)
        
        Args:
            punto: Diccionario con los valores del punto crítico
        
        Returns:
            String con la clasificación del punto
        """
        if self.hessiana_orlada is None:
            print("Error: Primero debe calcular la matriz Hessiana orlada")
            return "No clasificado"
        
        return self.clasificador.clasificar_punto_con_restricciones(
            self.hessiana_orlada, punto, self.variables,
            len(self.restricciones), "igualdad"
        )
    
    def analizar_puntos_con_restricciones(self):
        """
        Analiza y clasifica todos los puntos óptimos encontrados con restricciones
        """
        if not self.puntos_optimos:
            print("Error: Primero debe encontrar los puntos óptimos")
            return
        
        if self.hessiana_orlada is None:
            self.calcular_hessiana_orlada()
        
        self.clasificacion_puntos = []
        
        print("\nClasificando puntos óptimos usando el criterio de la segunda derivada con restricciones...")
        
        for i, punto in enumerate(self.puntos_optimos):
            if isinstance(punto, dict):
                clasificacion = self.clasificar_punto_con_restricciones(punto)
                self.clasificacion_puntos.append(clasificacion)
                print(f"\nPunto óptimo {i+1}: {clasificacion}")
            else:
                self.clasificacion_puntos.append("Formato de punto no soportado")
                print(f"\nPunto óptimo {i+1}: Formato no soportado para clasificación")
    
    def mostrar_puntos_optimos(self):
        """
        Muestra los puntos óptimos con sus clasificaciones y los valores de los multiplicadores
        """
        if not self.puntos_optimos:
            print("\nNo se encontraron puntos óptimos o no se han calculado aún.")
            return
        
        print("\n" + "="*80)
        print("PUNTOS ÓPTIMOS CON RESTRICCIONES Y CLASIFICACIÓN (MULTIPLICADORES DE LAGRANGE)")
        print("="*80)
        
        for i, solucion in enumerate(self.puntos_optimos, 1):
            print(f"\nSolución {i}:")
            
            if isinstance(solucion, dict):
                # Obtener clasificación
                clasificacion = None
                if i-1 < len(self.clasificacion_puntos):
                    clasificacion = self.clasificacion_puntos[i-1]
                
                # Usar el clasificador para mostrar información detallada
                self.clasificador.mostrar_punto_detallado(
                    solucion, self.variables, self.funcion_objetivo,
                    restricciones_igualdad=self.restricciones,
                    multiplicadores_lambda=self.multiplicadores,
                    clasificacion=clasificacion
                )
    
    def analisis_completo_con_restricciones(self, nombres_variables: List[str],
                                            expresion_funcion: str,
                                            expresiones_restricciones: List[str]):
        """
        Realiza el análisis completo de optimización con restricciones
        
        Args:
            nombres_variables: Lista de nombres de variables
            expresion_funcion: Expresión de la función objetivo
            expresiones_restricciones: Lista de expresiones de restricciones
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE OPTIMIZACIÓN CON RESTRICCIONES - MULTIPLICADORES DE LAGRANGE")
        print("="*70)
        
        # Paso 1: Definir variables
        self.definir_variables(nombres_variables)
        
        # Paso 2: Definir función objetivo
        if self.definir_funcion_objetivo(expresion_funcion) is None:
            return
        
        # Paso 3: Agregar restricciones
        for restriccion in expresiones_restricciones:
            if self.agregar_restriccion(restriccion) is None:
                return
        
        # Paso 4: Crear multiplicadores de Lagrange
        if self.crear_multiplicadores_lagrange() is None:
            return
        
        # Paso 5: Construir Lagrangiana
        if self.construir_lagrangiana() is None:
            return
        
        # Paso 6: Calcular gradiente
        if self.calcular_gradiente_lagrangiana() is None:
            return
        
        # Paso 7: Resolver sistema
        if self.resolver_sistema_lagrange() is None:
            return
        
        # Paso 8: Calcular matriz Hessiana orlada
        if self.calcular_hessiana_orlada() is None:
            print("Advertencia: No se pudo calcular la matriz Hessiana orlada")
        
        # Paso 9: Analizar y clasificar puntos óptimos
        self.analizar_puntos_con_restricciones()
        
        # Paso 10: Mostrar resultados
        self.mostrar_puntos_optimos()
    
    def generar_explicacion_didactica(self) -> str:
        """
        Genera una explicación didáctica paso a paso del problema de optimización con restricciones
        usando el método de Multiplicadores de Lagrange.
        
        Returns:
            String con la explicación didáctica completa
        """
        return self.formateador.generar_explicacion_completa(self, 'lagrange')