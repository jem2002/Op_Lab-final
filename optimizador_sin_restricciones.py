import sympy as sp
from sympy import symbols, diff, solve
from typing import List
from clasificador_puntos import ClasificadorPuntos
from formateador_didactico import FormateadorDidactico

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
            self.variables = [self.variables]
        print(f"Variables definidas: {self.variables}")
        return self.variables
    
    def definir_funcion_objetivo(self, expresion_str: str):
        """
        Define la función objetivo a partir de una expresión string
        
        Args:
            expresion_str: Expresión matemática como string (ej: 'x**2 + y**2 - 2*x*y')
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
        
        return self.clasificador.clasificar_punto_sin_restricciones(
            self.hessiana, punto_dict, self.variables
        )
    
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
            
            # Convertir punto a diccionario si es necesario
            if isinstance(punto, (list, tuple)):
                punto_dict = dict(zip(self.variables, punto))
            else:
                punto_dict = punto
            
            # Obtener clasificación
            clasificacion = None
            if i-1 < len(self.clasificacion_puntos):
                clasificacion = self.clasificacion_puntos[i-1]
            
            # Usar el clasificador para mostrar información detallada
            self.clasificador.mostrar_punto_detallado(
                punto_dict, self.variables, self.funcion_objetivo,
                clasificacion=clasificacion
            )
    
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
    
    def generar_explicacion_didactica(self) -> str:
        """
        Genera una explicación didáctica paso a paso del problema de optimización sin restricciones.
        
        Returns:
            String con la explicación didáctica completa
        """
        return self.formateador.generar_explicacion_completa(self, 'sin_restricciones')