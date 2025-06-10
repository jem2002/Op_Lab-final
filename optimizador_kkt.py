import sympy as sp
from sympy import symbols, diff, solve, pprint, latex, And, Or
import numpy as np
from typing import List, Dict, Tuple, Optional

class OptimizadorKKT:
    """
    Clase para resolver problemas de optimización no lineal con restricciones
    de desigualdad usando las Condiciones de Karush-Kuhn-Tucker (KKT)
    """
    
    def __init__(self):
        self.variables = []
        self.funcion_objetivo = None
        self.restricciones_igualdad = []  # h(x) = 0
        self.restricciones_desigualdad = []  # g(x) <= 0
        self.multiplicadores_lambda = []  # Para restricciones de igualdad
        self.multiplicadores_mu = []  # Para restricciones de desigualdad
        self.lagrangiana_kkt = None
        self.gradiente_lagrangiana = []
        self.hessiana_lagrangiana = None
        self.hessiana_orlada_kkt = None
        self.puntos_kkt = []
        self.condiciones_kkt = []
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
    
    def agregar_restriccion_igualdad(self, expresion_str: str):
        """
        Agrega una restricción de igualdad h(x) = 0
        
        Args:
            expresion_str: Expresión de la restricción (se asume = 0)
        """
        try:
            variables_dict = {str(var): var for var in self.variables}
            restriccion = sp.sympify(expresion_str, locals=variables_dict)
            self.restricciones_igualdad.append(restriccion)
            
            print(f"\nRestricción de igualdad {len(self.restricciones_igualdad)} agregada:")
            print(f"h_{len(self.restricciones_igualdad)}({', '.join(str(v) for v in self.variables)}) = {restriccion} = 0")
            
            return restriccion
            
        except Exception as e:
            print(f"Error al agregar restricción de igualdad: {e}")
            return None
    
    def agregar_restriccion_desigualdad(self, expresion_str: str):
        """
        Agrega una restricción de desigualdad g(x) <= 0
        
        Args:
            expresion_str: Expresión de la restricción (se asume <= 0)
        """
        try:
            variables_dict = {str(var): var for var in self.variables}
            restriccion = sp.sympify(expresion_str, locals=variables_dict)
            self.restricciones_desigualdad.append(restriccion)
            
            print(f"\nRestricción de desigualdad {len(self.restricciones_desigualdad)} agregada:")
            print(f"g_{len(self.restricciones_desigualdad)}({', '.join(str(v) for v in self.variables)}) = {restriccion} ≤ 0")
            
            return restriccion
            
        except Exception as e:
            print(f"Error al agregar restricción de desigualdad: {e}")
            return None
    
    def crear_multiplicadores_kkt(self):
        """
        Crea las variables de multiplicadores KKT (λ para igualdades, μ para desigualdades)
        """
        # Crear multiplicadores λ para restricciones de igualdad
        if self.restricciones_igualdad:
            nombres_lambda = [f'lambda_{i+1}' for i in range(len(self.restricciones_igualdad))]
            self.multiplicadores_lambda = symbols(nombres_lambda)
            # Asegurar que siempre sea una lista
            if not isinstance(self.multiplicadores_lambda, (list, tuple)):
                self.multiplicadores_lambda = [self.multiplicadores_lambda]
        
        # Crear multiplicadores μ para restricciones de desigualdad
        if self.restricciones_desigualdad:
            nombres_mu = [f'mu_{i+1}' for i in range(len(self.restricciones_desigualdad))]
            self.multiplicadores_mu = symbols(nombres_mu, positive=True)  # μ ≥ 0
            # Asegurar que siempre sea una lista
            if not isinstance(self.multiplicadores_mu, (list, tuple)):
                self.multiplicadores_mu = [self.multiplicadores_mu]
        
        print("\nMultiplicadores KKT creados:")
        if self.multiplicadores_lambda:
            print("  Multiplicadores λ (igualdades):")
            for i, lam in enumerate(self.multiplicadores_lambda):
                print(f"    λ_{i+1} = {lam}")
        
        if self.multiplicadores_mu:
            print("  Multiplicadores μ (desigualdades):")
            for i, mu in enumerate(self.multiplicadores_mu):
                print(f"    μ_{i+1} = {mu} ≥ 0")
        
        return self.multiplicadores_lambda, self.multiplicadores_mu
    
    def construir_lagrangiana_kkt(self):
        """
        Construye la función Lagrangiana KKT:
        L(x,λ,μ) = f(x) + Σ(λᵢ * hᵢ(x)) + Σ(μⱼ * gⱼ(x))
        """
        if self.funcion_objetivo is None:
            print("Error: Primero debe definir la función objetivo")
            return None
        
        if not self.restricciones_igualdad and not self.restricciones_desigualdad:
            print("Error: Debe definir al menos una restricción")
            return None
        
        if not self.multiplicadores_lambda and not self.multiplicadores_mu:
            self.crear_multiplicadores_kkt()
        
        # L(x,λ,μ) = f(x) + Σ(λᵢ * hᵢ(x)) + Σ(μⱼ * gⱼ(x))
        self.lagrangiana_kkt = self.funcion_objetivo
        
        print("\nConstruyendo función Lagrangiana KKT:")
        print(f"L = {self.funcion_objetivo}")
        
        # Agregar términos de restricciones de igualdad
        for i, (lam, restriccion) in enumerate(zip(self.multiplicadores_lambda, self.restricciones_igualdad)):
            # Convertir la restricción string a expresión simbólica si es necesario
            if isinstance(restriccion, str):
                restriccion_expr = sp.sympify(restriccion)
            else:
                restriccion_expr = restriccion
            
            termino = lam * restriccion_expr
            self.lagrangiana_kkt += termino
            print(f"    + {lam} * ({restriccion_expr})")
        
        # Agregar términos de restricciones de desigualdad
        for i, (mu, restriccion) in enumerate(zip(self.multiplicadores_mu, self.restricciones_desigualdad)):
            # Convertir la restricción string a expresión simbólica si es necesario
            if isinstance(restriccion, str):
                restriccion_expr = sp.sympify(restriccion)
            else:
                restriccion_expr = restriccion
            
            termino = mu * restriccion_expr
            self.lagrangiana_kkt += termino
            print(f"    + {mu} * ({restriccion_expr})")
        
        print(f"\nLagrangiana KKT completa:")
        todas_vars = list(self.variables) + list(self.multiplicadores_lambda) + list(self.multiplicadores_mu)
        print(f"L({', '.join(str(v) for v in todas_vars)}) = {self.lagrangiana_kkt}")
        
        return self.lagrangiana_kkt
    
    def calcular_gradiente_lagrangiana(self):
        """
        Calcula el gradiente de la Lagrangiana con respecto a todas las variables
        """
        if self.lagrangiana_kkt is None:
            print("Error: Primero debe construir la Lagrangiana KKT")
            return None
        
        self.gradiente_lagrangiana = []
        
        print("\nCalculando gradiente de la Lagrangiana KKT:")
        print("∇L = [")
        
        # Derivadas con respecto a las variables originales (condición de estacionariedad)
        for var in self.variables:
            derivada = diff(self.lagrangiana_kkt, var)
            self.gradiente_lagrangiana.append(derivada)
            print(f"  ∂L/∂{var} = {derivada}")
        
        print("]")
        
        return self.gradiente_lagrangiana
    
    def generar_condiciones_kkt(self):
        """
        Genera todas las condiciones KKT que deben cumplirse
        """
        if not self.gradiente_lagrangiana:
            print("Error: Primero debe calcular el gradiente de la Lagrangiana")
            return None
        
        self.condiciones_kkt = []
        
        print("\nCondiciones KKT:")
        print("="*50)
        
        # 1. Condiciones de estacionariedad: ∇L = 0
        print("1. Condiciones de estacionariedad:")
        for i, derivada in enumerate(self.gradiente_lagrangiana):
            condicion = sp.Eq(derivada, 0)
            self.condiciones_kkt.append(condicion)
            print(f"   ∂L/∂{self.variables[i]} = {derivada} = 0")
        
        # 2. Factibilidad primal: restricciones satisfechas
        print("\n2. Factibilidad primal:")
        # Restricciones de igualdad: h(x) = 0
        for i, restriccion in enumerate(self.restricciones_igualdad):
            condicion = sp.Eq(restriccion, 0)
            self.condiciones_kkt.append(condicion)
            print(f"   h_{i+1}(x) = {restriccion} = 0")
        
        # Restricciones de desigualdad: g(x) ≤ 0
        for i, restriccion in enumerate(self.restricciones_desigualdad):
            print(f"   g_{i+1}(x) = {restriccion} ≤ 0")
        
        # 3. Factibilidad dual: μ ≥ 0
        print("\n3. Factibilidad dual:")
        for i, mu in enumerate(self.multiplicadores_mu):
            print(f"   μ_{i+1} = {mu} ≥ 0")
        
        # 4. Holgura complementaria: μⱼ * gⱼ(x) = 0
        print("\n4. Holgura complementaria:")
        for i, (mu, restriccion) in enumerate(zip(self.multiplicadores_mu, self.restricciones_desigualdad)):
            condicion_complementaria = mu * restriccion
            print(f"   μ_{i+1} * g_{i+1}(x) = {mu} * ({restriccion}) = 0")
        
        return self.condiciones_kkt
    
    def resolver_sistema_kkt(self):
        """
        Resuelve el sistema de condiciones KKT
        """
        if not self.condiciones_kkt:
            self.generar_condiciones_kkt()
        
        todas_las_variables = list(self.variables) + list(self.multiplicadores_lambda) + list(self.multiplicadores_mu)
        
        print("\nResolviendo sistema KKT...")
        print("Sistema de ecuaciones principales:")
        
        # Resolver solo las condiciones de estacionariedad y factibilidad primal
        ecuaciones_principales = self.condiciones_kkt
        
        for i, eq in enumerate(ecuaciones_principales):
            print(f"  Ecuación {i+1}: {eq}")
        
        try:
            # Intentar resolver el sistema
            soluciones = solve(ecuaciones_principales, todas_las_variables)
            
            if isinstance(soluciones, dict):
                self.puntos_kkt = [soluciones]
            elif isinstance(soluciones, list):
                self.puntos_kkt = soluciones
            else:
                self.puntos_kkt = []
            
            # Filtrar soluciones que cumplan las condiciones de desigualdad
            soluciones_validas = []
            for solucion in self.puntos_kkt:
                if self.verificar_condiciones_kkt(solucion):
                    soluciones_validas.append(solucion)
            
            self.puntos_kkt = soluciones_validas
            return self.puntos_kkt
            
        except Exception as e:
            print(f"Error al resolver el sistema KKT: {e}")
            print("Nota: Los problemas KKT pueden ser complejos de resolver simbólicamente.")
            return None
    
    def verificar_condiciones_kkt(self, solucion: Dict) -> bool:
        """
        Verifica si una solución cumple todas las condiciones KKT
        
        Args:
            solucion: Diccionario con los valores de las variables
        
        Returns:
            True si cumple todas las condiciones KKT
        """
        try:
            # Verificar factibilidad dual: μ ≥ 0
            for mu in self.multiplicadores_mu:
                if mu in solucion:
                    valor_mu = solucion[mu]
                    if valor_mu < 0:
                        return False
        except Exception as e:
            print(f"Error al verificar factibilidad dual: {e}")
            return False
    
    def calcular_hessiana_lagrangiana_kkt(self):
        """
        Calcula la matriz Hessiana de la Lagrangiana KKT con respecto a las variables originales
        
        Returns:
            Matriz Hessiana de la Lagrangiana KKT
        """
        if self.lagrangiana_kkt is None:
            print("Error: Primero debe construir la Lagrangiana KKT")
            return None
        
        n = len(self.variables)
        self.hessiana_lagrangiana = sp.zeros(n, n)
        
        print("\nCalculando matriz Hessiana de la Lagrangiana KKT...")
        print("H_L = [")
        
        for i in range(n):
            for j in range(n):
                # Calcular la segunda derivada parcial ∂²L/∂xi∂xj
                segunda_derivada = diff(self.lagrangiana_kkt, self.variables[i], self.variables[j])
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
    
    def calcular_hessiana_orlada_kkt(self):
        """
        Calcula la matriz Hessiana orlada para problemas KKT
        (incluye restricciones de igualdad y desigualdad activas)
        
        Returns:
            Matriz Hessiana orlada KKT
        """
        if self.hessiana_lagrangiana is None:
            self.calcular_hessiana_lagrangiana_kkt()
        
        n = len(self.variables)  # número de variables
        m_eq = len(self.restricciones_igualdad)  # número de restricciones de igualdad
        m_ineq = len(self.restricciones_desigualdad)  # número de restricciones de desigualdad
        
        # Para KKT, consideramos todas las restricciones (activas e inactivas se determinan por μ)
        total_restricciones = m_eq + m_ineq
        
        # La matriz orlada tiene dimensión (n + total_restricciones) x (n + total_restricciones)
        self.hessiana_orlada_kkt = sp.zeros(n + total_restricciones, n + total_restricciones)
        
        print("\nCalculando matriz Hessiana orlada KKT...")
        
        # Bloque superior izquierdo: Hessiana de la Lagrangiana
        for i in range(n):
            for j in range(n):
                self.hessiana_orlada_kkt[i, j] = self.hessiana_lagrangiana[i, j]
        
        # Procesar restricciones de igualdad
        for k, restriccion in enumerate(self.restricciones_igualdad):
            # Bloque superior derecho: gradientes de restricciones de igualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, n + k] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de igualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[n + k, j] = grad_restriccion
        
        # Procesar restricciones de desigualdad
        for k, restriccion in enumerate(self.restricciones_desigualdad):
            idx = n + m_eq + k
            # Bloque superior derecho: gradientes de restricciones de desigualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, idx] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de desigualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[idx, j] = grad_restriccion
        
        # Bloque inferior derecho: ceros
        for i in range(total_restricciones):
            for j in range(total_restricciones):
                self.hessiana_orlada_kkt[n + i, n + j] = 0
        
        print(f"Matriz Hessiana orlada KKT de dimensión {n + total_restricciones}x{n + total_restricciones}:")
        print("H_orlada_KKT = [")
        for i in range(min(n + total_restricciones, 6)):  # Mostrar solo las primeras 6 filas para evitar salida muy larga
            fila = "  ["
            for j in range(min(n + total_restricciones, 6)):
                if j > 0:
                    fila += ", "
                fila += f"{self.hessiana_orlada_kkt[i, j]}"
            if n + total_restricciones > 6:
                fila += ", ..."
            fila += "]"
            print(fila)
        if n + total_restricciones > 6:
            print("  ...")
        print("]")
        
        return self.hessiana_orlada_kkt
    
    def identificar_restricciones_activas(self, punto):
        """
        Identifica qué restricciones de desigualdad están activas en un punto KKT
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            Lista de índices de restricciones activas
        """
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        restricciones_activas = []
        
        print("\nIdentificando restricciones activas:")
        
        # Verificar restricciones de desigualdad
        for i, restriccion in enumerate(self.restricciones_desigualdad):
            valor_restriccion = restriccion.subs(vars_originales)
            mu_correspondiente = self.multiplicadores_mu[i] if i < len(self.multiplicadores_mu) else None
            valor_mu = punto.get(mu_correspondiente, 0) if mu_correspondiente else 0
            
            try:
                val_rest_num = float(valor_restriccion)
                val_mu_num = float(valor_mu)
                
                # Una restricción está activa si g(x) ≈ 0 o μ > 0
                if abs(val_rest_num) < 1e-6 or val_mu_num > 1e-6:
                    restricciones_activas.append(i)
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (ACTIVA)")
                else:
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (inactiva)")
            
            except (ValueError, TypeError):
                print(f"  Restricción {i+1}: No se pudo evaluar numéricamente")
        
        return restricciones_activas
    
    def clasificar_punto_kkt(self, punto):
        """
        Clasifica un punto KKT usando el criterio de la segunda derivada
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            String con la clasificación del punto
        """
        if self.hessiana_orlada_kkt is None:
            print("Error: Primero debe calcular la matriz Hessiana orlada KKT")
            return "No clasificado"
        
        # Extraer solo las variables originales del punto
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        
        if not vars_originales:
            return "No se pudieron extraer las variables originales"
        
        try:
            # Identificar restricciones activas
            restricciones_activas = self.identificar_restricciones_activas(punto)
            
            # Evaluar la Hessiana orlada en el punto crítico
            hessiana_evaluada = self.hessiana_orlada_kkt.subs(vars_originales)
            
            n = len(self.variables)
            m_eq = len(self.restricciones_igualdad)
            m_activas = len(restricciones_activas)
            
            print(f"\nAnalizando punto KKT con {n} variables, {m_eq} restricciones de igualdad y {m_activas} restricciones activas...")
            
            # Para problemas KKT, el análisis depende de las restricciones activas
            # Construir la matriz Hessiana orlada reducida considerando solo restricciones activas
            
            if m_eq == 0 and m_activas == 0:
                # Sin restricciones activas: análisis estándar de la Hessiana
                hessiana_reducida = hessiana_evaluada[:n, :n]
                try:
                    autovalores = hessiana_reducida.eigenvals()
                    autovalores_numericos = [complex(val).real for val in autovalores.keys()]
                    
                    todos_positivos = all(val > 1e-10 for val in autovalores_numericos)
                    todos_negativos = all(val < -1e-10 for val in autovalores_numericos)
                    
                    if todos_positivos:
                        return f"Mínimo local (sin restricciones activas, autovalores > 0: {autovalores_numericos})"
                    elif todos_negativos:
                        return f"Máximo local (sin restricciones activas, autovalores < 0: {autovalores_numericos})"
                    else:
                        return f"Punto de silla (sin restricciones activas, autovalores mixtos: {autovalores_numericos})"
                
                except Exception as e:
                    return f"Error al analizar sin restricciones activas: {e}"
            
            else:
                # Con restricciones activas: análisis de la matriz orlada
                total_restricciones_activas = m_eq + m_activas
                
                # Calcular determinantes de submatrices relevantes
                determinantes = []
                
                for k in range(total_restricciones_activas + 1, n + total_restricciones_activas + 1):
                    if k <= hessiana_evaluada.rows:
                        submatriz = hessiana_evaluada[:k, :k]
                        try:
                            det = float(submatriz.det())
                            determinantes.append(det)
                            print(f"  Determinante de submatriz {k}x{k}: {det:.6f}")
                        except Exception as e:
                            print(f"  Error calculando determinante {k}x{k}: {e}")
                            determinantes.append(None)
                
                # Filtrar determinantes válidos
                dets_validos = [d for d in determinantes if d is not None]
                
                if not dets_validos:
                    return "No se pudieron evaluar los determinantes"
                
                # Análisis de los determinantes para clasificación KKT
                if len(dets_validos) == 1:
                    det = dets_validos[0]
                    signo_esperado = (-1) ** total_restricciones_activas
                    if det * signo_esperado > 0:
                        return f"Mínimo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    elif det * signo_esperado < 0:
                        return f"Máximo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    else:
                        return f"Criterio no concluyente KKT (det={det:.6f})"
                
                else:
                    # Análisis más complejo para múltiples determinantes
                    signos = [1 if d > 0 else -1 if d < 0 else 0 for d in dets_validos]
                    
                    # Verificar patrón para mínimo/máximo
                    patron_minimo = True
                    patron_maximo = True
                    
                    for i, signo in enumerate(signos):
                        signo_esperado_min = (-1) ** (total_restricciones_activas + i + 1)
                        signo_esperado_max = (-1) ** total_restricciones_activas
                        
                        if signo != signo_esperado_min:
                            patron_minimo = False
                        if signo != signo_esperado_max:
                            patron_maximo = False
                    
                    if patron_minimo:
                        return f"Mínimo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    elif patron_maximo:
                        return f"Máximo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    else:
                        return f"Punto de silla o indeterminado KKT ({m_activas} restricciones activas, dets={dets_validos})"
        
        except Exception as e:
            return f"Error en la clasificación KKT: {e}"
    
    def analizar_puntos_kkt(self):
        """
        Analiza y clasifica todos los puntos KKT encontrados
        """
        if not self.puntos_kkt:
            print("Error: Primero debe encontrar los puntos KKT")
            return
        
        if self.hessiana_orlada_kkt is None:
            self.calcular_hessiana_orlada_kkt()
        
        self.clasificacion_puntos = []
        
        print("\nClasificando puntos KKT usando el criterio de la segunda derivada...")
        
        for i, punto in enumerate(self.puntos_kkt):
            if isinstance(punto, dict):
                clasificacion = self.clasificar_punto_kkt(punto)
                self.clasificacion_puntos.append(clasificacion)
                print(f"\nPunto KKT {i+1}: {clasificacion}")
            else:
                self.clasificacion_puntos.append("Formato de punto no soportado")
                print(f"\nPunto KKT {i+1}: Formato no soportado para clasificación")
            
            # Verificar restricciones de desigualdad: g(x) ≤ 0
            vars_originales = {var: solucion.get(var, var) for var in self.variables}
            for restriccion in self.restricciones_desigualdad:
                valor_restriccion = restriccion.subs(vars_originales)
                if valor_restriccion > 0:
                    return False
    
    def calcular_hessiana_lagrangiana_kkt(self):
        """
        Calcula la matriz Hessiana de la Lagrangiana KKT con respecto a las variables originales
        
        Returns:
            Matriz Hessiana de la Lagrangiana KKT
        """
        if self.lagrangiana_kkt is None:
            print("Error: Primero debe construir la Lagrangiana KKT")
            return None
        
        n = len(self.variables)
        self.hessiana_lagrangiana = sp.zeros(n, n)
        
        print("\nCalculando matriz Hessiana de la Lagrangiana KKT...")
        print("H_L = [")
        
        for i in range(n):
            for j in range(n):
                # Calcular la segunda derivada parcial ∂²L/∂xi∂xj
                segunda_derivada = diff(self.lagrangiana_kkt, self.variables[i], self.variables[j])
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
    
    def calcular_hessiana_orlada_kkt(self):
        """
        Calcula la matriz Hessiana orlada para problemas KKT
        (incluye restricciones de igualdad y desigualdad activas)
        
        Returns:
            Matriz Hessiana orlada KKT
        """
        if self.hessiana_lagrangiana is None:
            self.calcular_hessiana_lagrangiana_kkt()
        
        n = len(self.variables)  # número de variables
        m_eq = len(self.restricciones_igualdad)  # número de restricciones de igualdad
        m_ineq = len(self.restricciones_desigualdad)  # número de restricciones de desigualdad
        
        # Para KKT, consideramos todas las restricciones (activas e inactivas se determinan por μ)
        total_restricciones = m_eq + m_ineq
        
        # La matriz orlada tiene dimensión (n + total_restricciones) x (n + total_restricciones)
        self.hessiana_orlada_kkt = sp.zeros(n + total_restricciones, n + total_restricciones)
        
        print("\nCalculando matriz Hessiana orlada KKT...")
        
        # Bloque superior izquierdo: Hessiana de la Lagrangiana
        for i in range(n):
            for j in range(n):
                self.hessiana_orlada_kkt[i, j] = self.hessiana_lagrangiana[i, j]
        
        # Procesar restricciones de igualdad
        for k, restriccion in enumerate(self.restricciones_igualdad):
            # Bloque superior derecho: gradientes de restricciones de igualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, n + k] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de igualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[n + k, j] = grad_restriccion
        
        # Procesar restricciones de desigualdad
        for k, restriccion in enumerate(self.restricciones_desigualdad):
            idx = n + m_eq + k
            # Bloque superior derecho: gradientes de restricciones de desigualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, idx] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de desigualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[idx, j] = grad_restriccion
        
        # Bloque inferior derecho: ceros
        for i in range(total_restricciones):
            for j in range(total_restricciones):
                self.hessiana_orlada_kkt[n + i, n + j] = 0
        
        print(f"Matriz Hessiana orlada KKT de dimensión {n + total_restricciones}x{n + total_restricciones}:")
        print("H_orlada_KKT = [")
        for i in range(min(n + total_restricciones, 6)):  # Mostrar solo las primeras 6 filas para evitar salida muy larga
            fila = "  ["
            for j in range(min(n + total_restricciones, 6)):
                if j > 0:
                    fila += ", "
                fila += f"{self.hessiana_orlada_kkt[i, j]}"
            if n + total_restricciones > 6:
                fila += ", ..."
            fila += "]"
            print(fila)
        if n + total_restricciones > 6:
            print("  ...")
        print("]")
        
        return self.hessiana_orlada_kkt
    
    def identificar_restricciones_activas(self, punto):
        """
        Identifica qué restricciones de desigualdad están activas en un punto KKT
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            Lista de índices de restricciones activas
        """
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        restricciones_activas = []
        
        print("\nIdentificando restricciones activas:")
        
        # Verificar restricciones de desigualdad
        for i, restriccion in enumerate(self.restricciones_desigualdad):
            valor_restriccion = restriccion.subs(vars_originales)
            mu_correspondiente = self.multiplicadores_mu[i] if i < len(self.multiplicadores_mu) else None
            valor_mu = punto.get(mu_correspondiente, 0) if mu_correspondiente else 0
            
            try:
                val_rest_num = float(valor_restriccion)
                val_mu_num = float(valor_mu)
                
                # Una restricción está activa si g(x) ≈ 0 o μ > 0
                if abs(val_rest_num) < 1e-6 or val_mu_num > 1e-6:
                    restricciones_activas.append(i)
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (ACTIVA)")
                else:
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (inactiva)")
            
            except (ValueError, TypeError):
                print(f"  Restricción {i+1}: No se pudo evaluar numéricamente")
        
        return restricciones_activas
    
    def clasificar_punto_kkt(self, punto):
        """
        Clasifica un punto KKT usando el criterio de la segunda derivada
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            String con la clasificación del punto
        """
        if self.hessiana_orlada_kkt is None:
            print("Error: Primero debe calcular la matriz Hessiana orlada KKT")
            return "No clasificado"
        
        # Extraer solo las variables originales del punto
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        
        if not vars_originales:
            return "No se pudieron extraer las variables originales"
        
        try:
            # Identificar restricciones activas
            restricciones_activas = self.identificar_restricciones_activas(punto)
            
            # Evaluar la Hessiana orlada en el punto crítico
            hessiana_evaluada = self.hessiana_orlada_kkt.subs(vars_originales)
            
            n = len(self.variables)
            m_eq = len(self.restricciones_igualdad)
            m_activas = len(restricciones_activas)
            
            print(f"\nAnalizando punto KKT con {n} variables, {m_eq} restricciones de igualdad y {m_activas} restricciones activas...")
            
            # Para problemas KKT, el análisis depende de las restricciones activas
            # Construir la matriz Hessiana orlada reducida considerando solo restricciones activas
            
            if m_eq == 0 and m_activas == 0:
                # Sin restricciones activas: análisis estándar de la Hessiana
                hessiana_reducida = hessiana_evaluada[:n, :n]
                try:
                    autovalores = hessiana_reducida.eigenvals()
                    autovalores_numericos = [complex(val).real for val in autovalores.keys()]
                    
                    todos_positivos = all(val > 1e-10 for val in autovalores_numericos)
                    todos_negativos = all(val < -1e-10 for val in autovalores_numericos)
                    
                    if todos_positivos:
                        return f"Mínimo local (sin restricciones activas, autovalores > 0: {autovalores_numericos})"
                    elif todos_negativos:
                        return f"Máximo local (sin restricciones activas, autovalores < 0: {autovalores_numericos})"
                    else:
                        return f"Punto de silla (sin restricciones activas, autovalores mixtos: {autovalores_numericos})"
                
                except Exception as e:
                    return f"Error al analizar sin restricciones activas: {e}"
            
            else:
                # Con restricciones activas: análisis de la matriz orlada
                total_restricciones_activas = m_eq + m_activas
                
                # Calcular determinantes de submatrices relevantes
                determinantes = []
                
                for k in range(total_restricciones_activas + 1, n + total_restricciones_activas + 1):
                    if k <= hessiana_evaluada.rows:
                        submatriz = hessiana_evaluada[:k, :k]
                        try:
                            det = float(submatriz.det())
                            determinantes.append(det)
                            print(f"  Determinante de submatriz {k}x{k}: {det:.6f}")
                        except Exception as e:
                            print(f"  Error calculando determinante {k}x{k}: {e}")
                            determinantes.append(None)
                
                # Filtrar determinantes válidos
                dets_validos = [d for d in determinantes if d is not None]
                
                if not dets_validos:
                    return "No se pudieron evaluar los determinantes"
                
                # Análisis de los determinantes para clasificación KKT
                if len(dets_validos) == 1:
                    det = dets_validos[0]
                    signo_esperado = (-1) ** total_restricciones_activas
                    if det * signo_esperado > 0:
                        return f"Mínimo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    elif det * signo_esperado < 0:
                        return f"Máximo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    else:
                        return f"Criterio no concluyente KKT (det={det:.6f})"
                
                else:
                    # Análisis más complejo para múltiples determinantes
                    signos = [1 if d > 0 else -1 if d < 0 else 0 for d in dets_validos]
                    
                    # Verificar patrón para mínimo/máximo
                    patron_minimo = True
                    patron_maximo = True
                    
                    for i, signo in enumerate(signos):
                        signo_esperado_min = (-1) ** (total_restricciones_activas + i + 1)
                        signo_esperado_max = (-1) ** total_restricciones_activas
                        
                        if signo != signo_esperado_min:
                            patron_minimo = False
                        if signo != signo_esperado_max:
                            patron_maximo = False
                    
                    if patron_minimo:
                        return f"Mínimo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    elif patron_maximo:
                        return f"Máximo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    else:
                        return f"Punto de silla o indeterminado KKT ({m_activas} restricciones activas, dets={dets_validos})"
        
        except Exception as e:
            return f"Error en la clasificación KKT: {e}"
    
    def analizar_puntos_kkt(self):
        """
        Analiza y clasifica todos los puntos KKT encontrados
        """
        if not self.puntos_kkt:
            print("Error: Primero debe encontrar los puntos KKT")
            return
        
        if self.hessiana_orlada_kkt is None:
            self.calcular_hessiana_orlada_kkt()
        
        self.clasificacion_puntos = []
        
        print("\nClasificando puntos KKT usando el criterio de la segunda derivada...")
        
        for i, punto in enumerate(self.puntos_kkt):
            if isinstance(punto, dict):
                clasificacion = self.clasificar_punto_kkt(punto)
                self.clasificacion_puntos.append(clasificacion)
                print(f"\nPunto KKT {i+1}: {clasificacion}")
            else:
                self.clasificacion_puntos.append("Formato de punto no soportado")
                print(f"\nPunto KKT {i+1}: Formato no soportado para clasificación")
            
            # Verificar holgura complementaria: μⱼ * gⱼ(x) = 0
            for mu, restriccion in zip(self.multiplicadores_mu, self.restricciones_desigualdad):
                if mu in solucion:
                    valor_mu = solucion[mu]
                    valor_restriccion = restriccion.subs(vars_originales)
                    producto = valor_mu * valor_restriccion
                    if abs(producto) > 1e-10:  # Tolerancia numérica
                        return False
    
    def calcular_hessiana_lagrangiana_kkt(self):
        """
        Calcula la matriz Hessiana de la Lagrangiana KKT con respecto a las variables originales
        
        Returns:
            Matriz Hessiana de la Lagrangiana KKT
        """
        if self.lagrangiana_kkt is None:
            print("Error: Primero debe construir la Lagrangiana KKT")
            return None
        
        n = len(self.variables)
        self.hessiana_lagrangiana = sp.zeros(n, n)
        
        print("\nCalculando matriz Hessiana de la Lagrangiana KKT...")
        print("H_L = [")
        
        for i in range(n):
            for j in range(n):
                # Calcular la segunda derivada parcial ∂²L/∂xi∂xj
                segunda_derivada = diff(self.lagrangiana_kkt, self.variables[i], self.variables[j])
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
    
    def calcular_hessiana_orlada_kkt(self):
        """
        Calcula la matriz Hessiana orlada para problemas KKT
        (incluye restricciones de igualdad y desigualdad activas)
        
        Returns:
            Matriz Hessiana orlada KKT
        """
        if self.hessiana_lagrangiana is None:
            self.calcular_hessiana_lagrangiana_kkt()
        
        n = len(self.variables)  # número de variables
        m_eq = len(self.restricciones_igualdad)  # número de restricciones de igualdad
        m_ineq = len(self.restricciones_desigualdad)  # número de restricciones de desigualdad
        
        # Para KKT, consideramos todas las restricciones (activas e inactivas se determinan por μ)
        total_restricciones = m_eq + m_ineq
        
        # La matriz orlada tiene dimensión (n + total_restricciones) x (n + total_restricciones)
        self.hessiana_orlada_kkt = sp.zeros(n + total_restricciones, n + total_restricciones)
        
        print("\nCalculando matriz Hessiana orlada KKT...")
        
        # Bloque superior izquierdo: Hessiana de la Lagrangiana
        for i in range(n):
            for j in range(n):
                self.hessiana_orlada_kkt[i, j] = self.hessiana_lagrangiana[i, j]
        
        # Procesar restricciones de igualdad
        for k, restriccion in enumerate(self.restricciones_igualdad):
            # Bloque superior derecho: gradientes de restricciones de igualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, n + k] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de igualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[n + k, j] = grad_restriccion
        
        # Procesar restricciones de desigualdad
        for k, restriccion in enumerate(self.restricciones_desigualdad):
            idx = n + m_eq + k
            # Bloque superior derecho: gradientes de restricciones de desigualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, idx] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de desigualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[idx, j] = grad_restriccion
        
        # Bloque inferior derecho: ceros
        for i in range(total_restricciones):
            for j in range(total_restricciones):
                self.hessiana_orlada_kkt[n + i, n + j] = 0
        
        print(f"Matriz Hessiana orlada KKT de dimensión {n + total_restricciones}x{n + total_restricciones}:")
        print("H_orlada_KKT = [")
        for i in range(min(n + total_restricciones, 6)):  # Mostrar solo las primeras 6 filas para evitar salida muy larga
            fila = "  ["
            for j in range(min(n + total_restricciones, 6)):
                if j > 0:
                    fila += ", "
                fila += f"{self.hessiana_orlada_kkt[i, j]}"
            if n + total_restricciones > 6:
                fila += ", ..."
            fila += "]"
            print(fila)
        if n + total_restricciones > 6:
            print("  ...")
        print("]")
        
        return self.hessiana_orlada_kkt
    
    def identificar_restricciones_activas(self, punto):
        """
        Identifica qué restricciones de desigualdad están activas en un punto KKT
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            Lista de índices de restricciones activas
        """
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        restricciones_activas = []
        
        print("\nIdentificando restricciones activas:")
        
        # Verificar restricciones de desigualdad
        for i, restriccion in enumerate(self.restricciones_desigualdad):
            valor_restriccion = restriccion.subs(vars_originales)
            mu_correspondiente = self.multiplicadores_mu[i] if i < len(self.multiplicadores_mu) else None
            valor_mu = punto.get(mu_correspondiente, 0) if mu_correspondiente else 0
            
            try:
                val_rest_num = float(valor_restriccion)
                val_mu_num = float(valor_mu)
                
                # Una restricción está activa si g(x) ≈ 0 o μ > 0
                if abs(val_rest_num) < 1e-6 or val_mu_num > 1e-6:
                    restricciones_activas.append(i)
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (ACTIVA)")
                else:
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (inactiva)")
            
            except (ValueError, TypeError):
                print(f"  Restricción {i+1}: No se pudo evaluar numéricamente")
        
        return restricciones_activas
    
    def clasificar_punto_kkt(self, punto):
        """
        Clasifica un punto KKT usando el criterio de la segunda derivada
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            String con la clasificación del punto
        """
        if self.hessiana_orlada_kkt is None:
            print("Error: Primero debe calcular la matriz Hessiana orlada KKT")
            return "No clasificado"
        
        # Extraer solo las variables originales del punto
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        
        if not vars_originales:
            return "No se pudieron extraer las variables originales"
        
        try:
            # Identificar restricciones activas
            restricciones_activas = self.identificar_restricciones_activas(punto)
            
            # Evaluar la Hessiana orlada en el punto crítico
            hessiana_evaluada = self.hessiana_orlada_kkt.subs(vars_originales)
            
            n = len(self.variables)
            m_eq = len(self.restricciones_igualdad)
            m_activas = len(restricciones_activas)
            
            print(f"\nAnalizando punto KKT con {n} variables, {m_eq} restricciones de igualdad y {m_activas} restricciones activas...")
            
            # Para problemas KKT, el análisis depende de las restricciones activas
            # Construir la matriz Hessiana orlada reducida considerando solo restricciones activas
            
            if m_eq == 0 and m_activas == 0:
                # Sin restricciones activas: análisis estándar de la Hessiana
                hessiana_reducida = hessiana_evaluada[:n, :n]
                try:
                    autovalores = hessiana_reducida.eigenvals()
                    autovalores_numericos = [complex(val).real for val in autovalores.keys()]
                    
                    todos_positivos = all(val > 1e-10 for val in autovalores_numericos)
                    todos_negativos = all(val < -1e-10 for val in autovalores_numericos)
                    
                    if todos_positivos:
                        return f"Mínimo local (sin restricciones activas, autovalores > 0: {autovalores_numericos})"
                    elif todos_negativos:
                        return f"Máximo local (sin restricciones activas, autovalores < 0: {autovalores_numericos})"
                    else:
                        return f"Punto de silla (sin restricciones activas, autovalores mixtos: {autovalores_numericos})"
                
                except Exception as e:
                    return f"Error al analizar sin restricciones activas: {e}"
            
            else:
                # Con restricciones activas: análisis de la matriz orlada
                total_restricciones_activas = m_eq + m_activas
                
                # Calcular determinantes de submatrices relevantes
                determinantes = []
                
                for k in range(total_restricciones_activas + 1, n + total_restricciones_activas + 1):
                    if k <= hessiana_evaluada.rows:
                        submatriz = hessiana_evaluada[:k, :k]
                        try:
                            det = float(submatriz.det())
                            determinantes.append(det)
                            print(f"  Determinante de submatriz {k}x{k}: {det:.6f}")
                        except Exception as e:
                            print(f"  Error calculando determinante {k}x{k}: {e}")
                            determinantes.append(None)
                
                # Filtrar determinantes válidos
                dets_validos = [d for d in determinantes if d is not None]
                
                if not dets_validos:
                    return "No se pudieron evaluar los determinantes"
                
                # Análisis de los determinantes para clasificación KKT
                if len(dets_validos) == 1:
                    det = dets_validos[0]
                    signo_esperado = (-1) ** total_restricciones_activas
                    if det * signo_esperado > 0:
                        return f"Mínimo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    elif det * signo_esperado < 0:
                        return f"Máximo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    else:
                        return f"Criterio no concluyente KKT (det={det:.6f})"
                
                else:
                    # Análisis más complejo para múltiples determinantes
                    signos = [1 if d > 0 else -1 if d < 0 else 0 for d in dets_validos]
                    
                    # Verificar patrón para mínimo/máximo
                    patron_minimo = True
                    patron_maximo = True
                    
                    for i, signo in enumerate(signos):
                        signo_esperado_min = (-1) ** (total_restricciones_activas + i + 1)
                        signo_esperado_max = (-1) ** total_restricciones_activas
                        
                        if signo != signo_esperado_min:
                            patron_minimo = False
                        if signo != signo_esperado_max:
                            patron_maximo = False
                    
                    if patron_minimo:
                        return f"Mínimo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    elif patron_maximo:
                        return f"Máximo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    else:
                        return f"Punto de silla o indeterminado KKT ({m_activas} restricciones activas, dets={dets_validos})"
        
        except Exception as e:
            return f"Error en la clasificación KKT: {e}"
    
    def analizar_puntos_kkt(self):
        """
        Analiza y clasifica todos los puntos KKT encontrados
        """
        if not self.puntos_kkt:
            print("Error: Primero debe encontrar los puntos KKT")
            return
        
        if self.hessiana_orlada_kkt is None:
            self.calcular_hessiana_orlada_kkt()
        
        self.clasificacion_puntos = []
        
        print("\nClasificando puntos KKT usando el criterio de la segunda derivada...")
        
        for i, punto in enumerate(self.puntos_kkt):
            if isinstance(punto, dict):
                clasificacion = self.clasificar_punto_kkt(punto)
                self.clasificacion_puntos.append(clasificacion)
                print(f"\nPunto KKT {i+1}: {clasificacion}")
            else:
                self.clasificacion_puntos.append("Formato de punto no soportado")
                print(f"\nPunto KKT {i+1}: Formato no soportado para clasificación")
            
            return True
    
    def calcular_hessiana_lagrangiana_kkt(self):
        """
        Calcula la matriz Hessiana de la Lagrangiana KKT con respecto a las variables originales
        
        Returns:
            Matriz Hessiana de la Lagrangiana KKT
        """
        if self.lagrangiana_kkt is None:
            print("Error: Primero debe construir la Lagrangiana KKT")
            return None
        
        n = len(self.variables)
        self.hessiana_lagrangiana = sp.zeros(n, n)
        
        print("\nCalculando matriz Hessiana de la Lagrangiana KKT...")
        print("H_L = [")
        
        for i in range(n):
            for j in range(n):
                # Calcular la segunda derivada parcial ∂²L/∂xi∂xj
                segunda_derivada = diff(self.lagrangiana_kkt, self.variables[i], self.variables[j])
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
    
    def calcular_hessiana_orlada_kkt(self):
        """
        Calcula la matriz Hessiana orlada para problemas KKT
        (incluye restricciones de igualdad y desigualdad activas)
        
        Returns:
            Matriz Hessiana orlada KKT
        """
        if self.hessiana_lagrangiana is None:
            self.calcular_hessiana_lagrangiana_kkt()
        
        n = len(self.variables)  # número de variables
        m_eq = len(self.restricciones_igualdad)  # número de restricciones de igualdad
        m_ineq = len(self.restricciones_desigualdad)  # número de restricciones de desigualdad
        
        # Para KKT, consideramos todas las restricciones (activas e inactivas se determinan por μ)
        total_restricciones = m_eq + m_ineq
        
        # La matriz orlada tiene dimensión (n + total_restricciones) x (n + total_restricciones)
        self.hessiana_orlada_kkt = sp.zeros(n + total_restricciones, n + total_restricciones)
        
        print("\nCalculando matriz Hessiana orlada KKT...")
        
        # Bloque superior izquierdo: Hessiana de la Lagrangiana
        for i in range(n):
            for j in range(n):
                self.hessiana_orlada_kkt[i, j] = self.hessiana_lagrangiana[i, j]
        
        # Procesar restricciones de igualdad
        for k, restriccion in enumerate(self.restricciones_igualdad):
            # Bloque superior derecho: gradientes de restricciones de igualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, n + k] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de igualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[n + k, j] = grad_restriccion
        
        # Procesar restricciones de desigualdad
        for k, restriccion in enumerate(self.restricciones_desigualdad):
            idx = n + m_eq + k
            # Bloque superior derecho: gradientes de restricciones de desigualdad
            for i in range(n):
                grad_restriccion = diff(restriccion, self.variables[i])
                self.hessiana_orlada_kkt[i, idx] = grad_restriccion
            
            # Bloque inferior izquierdo: gradientes de restricciones de desigualdad (transpuesto)
            for j in range(n):
                grad_restriccion = diff(restriccion, self.variables[j])
                self.hessiana_orlada_kkt[idx, j] = grad_restriccion
        
        # Bloque inferior derecho: ceros
        for i in range(total_restricciones):
            for j in range(total_restricciones):
                self.hessiana_orlada_kkt[n + i, n + j] = 0
        
        print(f"Matriz Hessiana orlada KKT de dimensión {n + total_restricciones}x{n + total_restricciones}:")
        print("H_orlada_KKT = [")
        for i in range(min(n + total_restricciones, 6)):  # Mostrar solo las primeras 6 filas para evitar salida muy larga
            fila = "  ["
            for j in range(min(n + total_restricciones, 6)):
                if j > 0:
                    fila += ", "
                fila += f"{self.hessiana_orlada_kkt[i, j]}"
            if n + total_restricciones > 6:
                fila += ", ..."
            fila += "]"
            print(fila)
        if n + total_restricciones > 6:
            print("  ...")
        print("]")
        
        return self.hessiana_orlada_kkt
    
    def identificar_restricciones_activas(self, punto):
        """
        Identifica qué restricciones de desigualdad están activas en un punto KKT
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            Lista de índices de restricciones activas
        """
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        restricciones_activas = []
        
        print("\nIdentificando restricciones activas:")
        
        # Verificar restricciones de desigualdad
        for i, restriccion in enumerate(self.restricciones_desigualdad):
            valor_restriccion = restriccion.subs(vars_originales)
            mu_correspondiente = self.multiplicadores_mu[i] if i < len(self.multiplicadores_mu) else None
            valor_mu = punto.get(mu_correspondiente, 0) if mu_correspondiente else 0
            
            try:
                val_rest_num = float(valor_restriccion)
                val_mu_num = float(valor_mu)
                
                # Una restricción está activa si g(x) ≈ 0 o μ > 0
                if abs(val_rest_num) < 1e-6 or val_mu_num > 1e-6:
                    restricciones_activas.append(i)
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (ACTIVA)")
                else:
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (inactiva)")
            
            except (ValueError, TypeError):
                print(f"  Restricción {i+1}: No se pudo evaluar numéricamente")
        
        return restricciones_activas
    
    def clasificar_punto_kkt(self, punto):
        """
        Clasifica un punto KKT usando el criterio de la segunda derivada
        
        Args:
            punto: Diccionario con los valores del punto KKT
        
        Returns:
            String con la clasificación del punto
        """
        if self.hessiana_orlada_kkt is None:
            print("Error: Primero debe calcular la matriz Hessiana orlada KKT")
            return "No clasificado"
        
        # Extraer solo las variables originales del punto
        vars_originales = {var: punto.get(var, var) for var in self.variables if var in punto}
        
        if not vars_originales:
            return "No se pudieron extraer las variables originales"
        
        try:
            # Identificar restricciones activas
            restricciones_activas = self.identificar_restricciones_activas(punto)
            
            # Evaluar la Hessiana orlada en el punto crítico
            hessiana_evaluada = self.hessiana_orlada_kkt.subs(vars_originales)
            
            n = len(self.variables)
            m_eq = len(self.restricciones_igualdad)
            m_activas = len(restricciones_activas)
            
            print(f"\nAnalizando punto KKT con {n} variables, {m_eq} restricciones de igualdad y {m_activas} restricciones activas...")
            
            # Para problemas KKT, el análisis depende de las restricciones activas
            # Construir la matriz Hessiana orlada reducida considerando solo restricciones activas
            
            if m_eq == 0 and m_activas == 0:
                # Sin restricciones activas: análisis estándar de la Hessiana
                hessiana_reducida = hessiana_evaluada[:n, :n]
                try:
                    autovalores = hessiana_reducida.eigenvals()
                    autovalores_numericos = [complex(val).real for val in autovalores.keys()]
                    
                    todos_positivos = all(val > 1e-10 for val in autovalores_numericos)
                    todos_negativos = all(val < -1e-10 for val in autovalores_numericos)
                    
                    if todos_positivos:
                        return f"Mínimo local (sin restricciones activas, autovalores > 0: {autovalores_numericos})"
                    elif todos_negativos:
                        return f"Máximo local (sin restricciones activas, autovalores < 0: {autovalores_numericos})"
                    else:
                        return f"Punto de silla (sin restricciones activas, autovalores mixtos: {autovalores_numericos})"
                
                except Exception as e:
                    return f"Error al analizar sin restricciones activas: {e}"
            
            else:
                # Con restricciones activas: análisis de la matriz orlada
                total_restricciones_activas = m_eq + m_activas
                
                # Calcular determinantes de submatrices relevantes
                determinantes = []
                
                for k in range(total_restricciones_activas + 1, n + total_restricciones_activas + 1):
                    if k <= hessiana_evaluada.rows:
                        submatriz = hessiana_evaluada[:k, :k]
                        try:
                            det = float(submatriz.det())
                            determinantes.append(det)
                            print(f"  Determinante de submatriz {k}x{k}: {det:.6f}")
                        except Exception as e:
                            print(f"  Error calculando determinante {k}x{k}: {e}")
                            determinantes.append(None)
                
                # Filtrar determinantes válidos
                dets_validos = [d for d in determinantes if d is not None]
                
                if not dets_validos:
                    return "No se pudieron evaluar los determinantes"
                
                # Análisis de los determinantes para clasificación KKT
                if len(dets_validos) == 1:
                    det = dets_validos[0]
                    signo_esperado = (-1) ** total_restricciones_activas
                    if det * signo_esperado > 0:
                        return f"Mínimo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    elif det * signo_esperado < 0:
                        return f"Máximo local condicionado KKT (det={det:.6f}, {m_activas} restricciones activas)"
                    else:
                        return f"Criterio no concluyente KKT (det={det:.6f})"
                
                else:
                    # Análisis más complejo para múltiples determinantes
                    signos = [1 if d > 0 else -1 if d < 0 else 0 for d in dets_validos]
                    
                    # Verificar patrón para mínimo/máximo
                    patron_minimo = True
                    patron_maximo = True
                    
                    for i, signo in enumerate(signos):
                        signo_esperado_min = (-1) ** (total_restricciones_activas + i + 1)
                        signo_esperado_max = (-1) ** total_restricciones_activas
                        
                        if signo != signo_esperado_min:
                            patron_minimo = False
                        if signo != signo_esperado_max:
                            patron_maximo = False
                    
                    if patron_minimo:
                        return f"Mínimo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    elif patron_maximo:
                        return f"Máximo local condicionado KKT ({m_activas} restricciones activas, dets={dets_validos})"
                    else:
                        return f"Punto de silla o indeterminado KKT ({m_activas} restricciones activas, dets={dets_validos})"
        
        except Exception as e:
            return f"Error en la clasificación KKT: {e}"
    
    def analizar_puntos_kkt(self):
        """
        Analiza y clasifica todos los puntos KKT encontrados
        """
        if not self.puntos_kkt:
            print("Error: Primero debe encontrar los puntos KKT")
            return
        
        if self.hessiana_orlada_kkt is None:
            self.calcular_hessiana_orlada_kkt()
        
        self.clasificacion_puntos = []
        
        print("\nClasificando puntos KKT usando el criterio de la segunda derivada...")
        
        for i, punto in enumerate(self.puntos_kkt):
            if isinstance(punto, dict):
                clasificacion = self.clasificar_punto_kkt(punto)
                self.clasificacion_puntos.append(clasificacion)
                print(f"\nPunto KKT {i+1}: {clasificacion}")
            else:
                self.clasificacion_puntos.append("Formato de punto no soportado")
                print(f"\nPunto KKT {i+1}: Formato no soportado para clasificación")
    
    def mostrar_puntos_kkt(self):
        """
        Muestra los puntos que satisfacen las condiciones KKT
        """
        if not self.puntos_kkt:
            print("\nNo se encontraron puntos que satisfagan las condiciones KKT.")
            return
        
        print("\n" + "="*70)
        print("PUNTOS QUE SATISFACEN LAS CONDICIONES KKT")
        print("="*70)
        
        for i, solucion in enumerate(self.puntos_kkt, 1):
            print(f"\nSolución KKT {i}:")
            
            if isinstance(solucion, dict):
                # Separar variables originales y multiplicadores
                vars_originales = {}
                multiplicadores_vals = {}
                
                for var, valor in solucion.items():
                    if var in self.variables:
                        vars_originales[var] = valor
                    elif var in self.multiplicadores_lambda or var in self.multiplicadores_mu:
                        multiplicadores_vals[var] = valor
                
                # Mostrar variables originales
                print("  Variables:")
                for var, valor in vars_originales.items():
                    print(f"    {var} = {valor}")
                
                # Mostrar multiplicadores
                if multiplicadores_vals:
                    print("  Multiplicadores KKT:")
                    for mult, valor in multiplicadores_vals.items():
                        print(f"    {mult} = {valor}")
                
                # Evaluar función objetivo
                if vars_originales:
                    valor_funcion = self.funcion_objetivo.subs(vars_originales)
                    print(f"  Valor de la función objetivo: f = {valor_funcion}")
                
                # Verificar restricciones
                print("  Verificación de restricciones:")
                
                # Restricciones de igualdad
                for j, restriccion in enumerate(self.restricciones_igualdad):
                    valor_restriccion = restriccion.subs(vars_originales)
                    print(f"    h_{j+1} = {valor_restriccion} (debe ser = 0)")
                
                # Restricciones de desigualdad
                for j, restriccion in enumerate(self.restricciones_desigualdad):
                    valor_restriccion = restriccion.subs(vars_originales)
                    print(f"    g_{j+1} = {valor_restriccion} (debe ser ≤ 0)")
                
                # Verificar holgura complementaria
                print("  Holgura complementaria:")
                for j, (mu, restriccion) in enumerate(zip(self.multiplicadores_mu, self.restricciones_desigualdad)):
                    if mu in multiplicadores_vals:
                        valor_mu = multiplicadores_vals[mu]
                        valor_restriccion = restriccion.subs(vars_originales)
                        producto = valor_mu * valor_restriccion
                        print(f"    μ_{j+1} * g_{j+1} = {valor_mu} * {valor_restriccion} = {producto} (debe ser = 0)")
                
                # Mostrar clasificación si está disponible
                if i-1 < len(self.clasificacion_puntos):
                    print(f"  Clasificación: {self.clasificacion_puntos[i-1]}")
                else:
                    print("  Clasificación: No calculada")
    
    def analisis_completo_kkt(self, nombres_variables: List[str], 
                             expresion_funcion: str,
                             restricciones_igualdad: List[str] = None,
                             restricciones_desigualdad: List[str] = None):
        """
        Realiza el análisis completo usando condiciones KKT
        
        Args:
            nombres_variables: Lista de nombres de variables
            expresion_funcion: Expresión de la función objetivo
            restricciones_igualdad: Lista de restricciones de igualdad
            restricciones_desigualdad: Lista de restricciones de desigualdad
        """
        print("\n" + "="*80)
        print("ANÁLISIS DE OPTIMIZACIÓN CON RESTRICCIONES - CONDICIONES KKT")
        print("="*80)
        
        # Paso 1: Definir variables
        self.definir_variables(nombres_variables)
        
        # Paso 2: Definir función objetivo
        if self.definir_funcion_objetivo(expresion_funcion) is None:
            return
        
        # Paso 3: Agregar restricciones de igualdad
        if restricciones_igualdad:
            for restriccion in restricciones_igualdad:
                if self.agregar_restriccion_igualdad(restriccion) is None:
                    return
        
        # Paso 4: Agregar restricciones de desigualdad
        if restricciones_desigualdad:
            for restriccion in restricciones_desigualdad:
                if self.agregar_restriccion_desigualdad(restriccion) is None:
                    return
        
        # Verificar que hay al menos una restricción
        if not self.restricciones_igualdad and not self.restricciones_desigualdad:
            print("Error: Debe definir al menos una restricción para usar KKT.")
            return
        
        # Paso 5: Crear multiplicadores KKT
        if self.crear_multiplicadores_kkt() is None:
            return
        
        # Paso 6: Construir Lagrangiana KKT
        if self.construir_lagrangiana_kkt() is None:
            return
        
        # Paso 7: Calcular gradiente
        if self.calcular_gradiente_lagrangiana() is None:
            return
        
        # Paso 8: Generar condiciones KKT
        if self.generar_condiciones_kkt() is None:
            return
        
        # Paso 9: Resolver sistema KKT
        if self.resolver_sistema_kkt() is None:
            print("\nNota: El sistema KKT puede requerir análisis manual adicional.")
            print("Las condiciones KKT son necesarias pero no suficientes para optimalidad.")
            return
        
        # Paso 10: Calcular matriz Hessiana orlada KKT
        if self.calcular_hessiana_orlada_kkt() is None:
            print("Advertencia: No se pudo calcular la matriz Hessiana orlada KKT")
        
        # Paso 11: Analizar y clasificar puntos KKT
        self.analizar_puntos_kkt()
        
        # Paso 12: Mostrar resultados
        self.mostrar_puntos_kkt()

def mostrar_ejemplos_kkt(optimizador):
    """
    Muestra ejemplos predefinidos con condiciones KKT
    """
    ejemplos = [
        {
            "nombre": "Optimización con restricción de desigualdad simple",
            "variables": ["x", "y"],
            "funcion": "x**2 + y**2",
            "restricciones_igualdad": [],
            "restricciones_desigualdad": ["x + y - 1"],
            "descripcion": "Minimizar x²+y² sujeto a x+y≤1"
        },
        {
            "nombre": "Problema con restricción de caja",
            "variables": ["x", "y"],
            "funcion": "-(x*y)",
            "restricciones_igualdad": [],
            "restricciones_desigualdad": ["-x", "-y", "x - 2", "y - 2"],
            "descripcion": "Maximizar xy en [0,2]×[0,2]"
        },
        {
            "nombre": "Optimización con restricciones mixtas",
            "variables": ["x", "y"],
            "funcion": "x**2 + y**2",
            "restricciones_igualdad": ["x + y - 2"],
            "restricciones_desigualdad": ["-x", "-y"],
            "descripcion": "Minimizar x²+y² con x+y=2, x≥0, y≥0"
        },
        {
            "nombre": "Problema de programación cuadrática",
            "variables": ["x", "y"],
            "funcion": "x**2 + y**2 - 2*x - 4*y",
            "restricciones_igualdad": [],
            "restricciones_desigualdad": ["-x", "-y", "x + 2*y - 3"],
            "descripcion": "Minimizar función cuadrática con restricciones lineales"
        }
    ]
    
    print("\nEjemplos CON restricciones KKT disponibles:")
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"{i}. {ejemplo['nombre']}")
        print(f"   Variables: {ejemplo['variables']}")
        print(f"   Función objetivo: {ejemplo['funcion']}")
        if ejemplo['restricciones_igualdad']:
            print(f"   Restricciones de igualdad: {ejemplo['restricciones_igualdad']}")
        if ejemplo['restricciones_desigualdad']:
            print(f"   Restricciones de desigualdad: {ejemplo['restricciones_desigualdad']}")
        print(f"   Descripción: {ejemplo['descripcion']}\n")
    
    try:
        seleccion = int(input("Seleccione un ejemplo (1-4): ")) - 1
        
        if 0 <= seleccion < len(ejemplos):
            ejemplo = ejemplos[seleccion]
            print(f"\nEjecutando: {ejemplo['nombre']}")
            optimizador.analisis_completo_kkt(
                ejemplo['variables'], 
                ejemplo['funcion'], 
                ejemplo['restricciones_igualdad'],
                ejemplo['restricciones_desigualdad']
            )
        else:
            print("Selección no válida.")
            
    except ValueError:
        print("Por favor, ingrese un número válido.")

def analisis_kkt_interactivo(optimizador):
    """
    Realiza un análisis KKT de forma interactiva
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
        print("  - x*y")
        print("  - x**2 + 2*y**2 - 4*x - 6*y")
        
        expresion_objetivo = input("\nFunción objetivo: ").strip()
        
        # Solicitar restricciones de igualdad
        restricciones_igualdad = []
        print("\nIngrese las restricciones de IGUALDAD (h(x,y,...) = 0).")
        print("Ingrese solo la parte izquierda de la ecuación (se asume = 0).")
        print("Ejemplos:")
        print("  - x + y - 1  (para x + y = 1)")
        print("  - x**2 + y**2 - 4  (para x² + y² = 4)")
        print("Presione Enter sin escribir nada si no hay restricciones de igualdad.")
        
        while True:
            restriccion = input(f"\nRestricción de igualdad {len(restricciones_igualdad)+1} (o Enter para continuar): ").strip()
            if not restriccion:
                break
            restricciones_igualdad.append(restriccion)
        
        # Solicitar restricciones de desigualdad
        restricciones_desigualdad = []
        print("\nIngrese las restricciones de DESIGUALDAD (g(x,y,...) ≤ 0).")
        print("Ingrese solo la parte izquierda de la desigualdad (se asume ≤ 0).")
        print("Ejemplos:")
        print("  - x + y - 1  (para x + y ≤ 1)")
        print("  - -x  (para x ≥ 0, equivale a -x ≤ 0)")
        print("  - x**2 + y**2 - 4  (para x² + y² ≤ 4)")
        
        while True:
            restriccion = input(f"\nRestricción de desigualdad {len(restricciones_desigualdad)+1} (o 'fin' para terminar): ").strip()
            if restriccion.lower() == 'fin':
                break
            if restriccion:
                restricciones_desigualdad.append(restriccion)
        
        # Verificar que hay al menos una restricción
        if not restricciones_igualdad and not restricciones_desigualdad:
            print("Error: Debe definir al menos una restricción para usar condiciones KKT.")
            return
        
        # Realizar análisis
        optimizador.analisis_completo_kkt(nombres_variables, expresion_objetivo, 
                                         restricciones_igualdad, restricciones_desigualdad)
        
    except Exception as e:
        print(f"Error durante el análisis KKT: {e}")