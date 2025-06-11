import sympy as sp
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class ClasificadorPuntos:
    """
    Clase modular para clasificar y verificar puntos críticos/óptimos
    en problemas de optimización no lineal.
    
    Centraliza la lógica común de clasificación de puntos para:
    - Optimización sin restricciones
    - Optimización con restricciones de igualdad (Lagrange)
    - Optimización con restricciones de desigualdad (KKT)
    """
    
    def __init__(self):
        self.tolerancia = 1e-6
    
    def extraer_variables_originales(self, punto: Dict, variables: List) -> Dict:
        """
        Extrae solo las variables originales de un punto que puede contener multiplicadores de Lagrange o KKT.
        
        Args:
            punto: Diccionario con todas las variables
            variables: Lista de variables originales del problema
        
        Returns:
            Diccionario con solo las variables originales
        """
        if isinstance(punto, dict):
            return {var: punto.get(var, var) for var in variables if var in punto}
        elif isinstance(punto, (list, tuple)):
            return dict(zip(variables, punto))
        else:
            return {}
    
    def evaluar_funcion_en_punto(self, funcion, punto: Dict) -> Union[float, sp.Basic]:
        """
        Evalúa una función simbólica en un punto dado.
        
        Args:
            funcion: Función simbólica de SymPy
            punto: Diccionario con valores de las variables
        
        Returns:
            Valor de la función en el punto
        """
        try:
            return funcion.subs(punto)
        except Exception as e:
            print(f"Error al evaluar función en punto: {e}")
            return None
    
    def verificar_restricciones_igualdad(self, restricciones: List, punto: Dict) -> List[Tuple[int, float]]:
        """
        Verifica qué tan bien se satisfacen las restricciones de igualdad en un punto.
        
        Args:
            restricciones: Lista de restricciones de igualdad
            punto: Diccionario con valores de las variables
        
        Returns:
            Lista de tuplas (índice, valor_restricción)
        """
        resultados = []
        for i, restriccion in enumerate(restricciones):
            try:
                valor = float(restriccion.subs(punto))
                resultados.append((i, valor))
            except (ValueError, TypeError):
                resultados.append((i, None))
        return resultados
    
    def verificar_restricciones_desigualdad(self, restricciones: List, punto: Dict) -> List[Tuple[int, float, bool]]:
        """
        Verifica qué restricciones de desigualdad están activas en un punto.
        
        Args:
            restricciones: Lista de restricciones de desigualdad (g(x) <= 0)
            punto: Diccionario con valores de las variables
        
        Returns:
            Lista de tuplas (índice, valor_restricción, es_activa)
        """
        resultados = []
        for i, restriccion in enumerate(restricciones):
            try:
                valor = float(restriccion.subs(punto))
                es_activa = abs(valor) < self.tolerancia
                resultados.append((i, valor, es_activa))
            except (ValueError, TypeError):
                resultados.append((i, None, False))
        return resultados
    
    def clasificar_punto_sin_restricciones(self, hessiana, punto: Dict, variables: List) -> str:
        """
        Clasifica un punto crítico sin restricciones usando la matriz Hessiana.
        
        Args:
            hessiana: Matriz Hessiana simbólica
            punto: Diccionario con valores del punto crítico
            variables: Lista de variables del problema
        
        Returns:
            String con la clasificación del punto
        """
        try:
            # Evaluar la Hessiana en el punto crítico
            hessiana_evaluada = hessiana.subs(punto)
            n = len(variables)
            
            if n == 1:
                # Caso unidimensional
                segunda_derivada = float(hessiana_evaluada[0, 0])
                if segunda_derivada > self.tolerancia:
                    return "Mínimo local"
                elif segunda_derivada < -self.tolerancia:
                    return "Máximo local"
                else:
                    return "Criterio no concluyente (segunda derivada ≈ 0)"
            
            elif n == 2:
                # Caso bidimensional: usar determinante y traza
                h11 = float(hessiana_evaluada[0, 0])
                h12 = float(hessiana_evaluada[0, 1])
                h21 = float(hessiana_evaluada[1, 0])
                h22 = float(hessiana_evaluada[1, 1])
                
                determinante = h11 * h22 - h12 * h21
                traza = h11 + h22
                
                if determinante > self.tolerancia:
                    if traza > self.tolerancia:
                        return f"Mínimo local (det={determinante:.4f} > 0, tr={traza:.4f} > 0)"
                    else:
                        return f"Máximo local (det={determinante:.4f} > 0, tr={traza:.4f} < 0)"
                elif determinante < -self.tolerancia:
                    return f"Punto de silla (det={determinante:.4f} < 0)"
                else:
                    return f"Criterio no concluyente (det={determinante:.4f} ≈ 0)"
            
            else:
                # Caso multidimensional: verificar definitud usando autovalores
                autovalores = hessiana_evaluada.eigenvals()
                autovalores_numericos = [complex(val).real for val in autovalores.keys()]
                
                todos_positivos = all(val > self.tolerancia for val in autovalores_numericos)
                todos_negativos = all(val < -self.tolerancia for val in autovalores_numericos)
                
                if todos_positivos:
                    return f"Mínimo local (todos los autovalores > 0: {[round(v, 4) for v in autovalores_numericos]})"
                elif todos_negativos:
                    return f"Máximo local (todos los autovalores < 0: {[round(v, 4) for v in autovalores_numericos]})"
                else:
                    return f"Punto de silla (autovalores mixtos: {[round(v, 4) for v in autovalores_numericos]})"
        
        except Exception as e:
            return f"Error al clasificar punto: {e}"
    
    def clasificar_punto_con_restricciones(self, hessiana_orlada, punto: Dict,
                                        variables: List, num_restricciones: int,
                                        tipo_restricciones: str = "igualdad") -> str:
        """
        Clasifica un punto con restricciones usando la matriz Hessiana orlada.
        
        Args:
            hessiana_orlada: Matriz Hessiana orlada
            punto: Diccionario con valores del punto
            variables: Lista de variables originales
            num_restricciones: Número de restricciones
            tipo_restricciones: "igualdad", "kkt", o "mixto"
        
        Returns:
            String con la clasificación del punto
        """
        try:
            vars_originales = self.extraer_variables_originales(punto, variables)
            hessiana_evaluada = hessiana_orlada.subs(vars_originales)
            
            n = len(variables)
            m = num_restricciones
            
            print(f"\nAnalizando punto con {n} variables y {m} restricciones ({tipo_restricciones})...")
            
            # Calcular determinantes de submatrices relevantes
            determinantes = []
            
            for k in range(m + 1, n + m + 1):
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
            
            # Análisis de los determinantes para clasificación
            if len(dets_validos) == 1:
                det = dets_validos[0]
                signo_esperado = (-1) ** m
                if det * signo_esperado > self.tolerancia:
                    return f"Mínimo local condicionado (det={det:.6f})"
                elif det * signo_esperado < -self.tolerancia:
                    return f"Máximo local condicionado (det={det:.6f})"
                else:
                    return f"Criterio no concluyente (det={det:.6f})"
            
            else:
                # Análisis más complejo para múltiples determinantes
                signos = [1 if d > self.tolerancia else -1 if d < -self.tolerancia else 0 for d in dets_validos]
                
                # Verificar patrón alternante
                patron_minimo = True
                patron_maximo = True
                
                for i, signo in enumerate(signos):
                    signo_esperado_min = (-1) ** (m + i + 1)
                    signo_esperado_max = (-1) ** m
                    
                    if signo != signo_esperado_min:
                        patron_minimo = False
                    if signo != signo_esperado_max:
                        patron_maximo = False
                
                if patron_minimo:
                    return f"Mínimo local condicionado (dets={[round(d, 4) for d in dets_validos]})"
                elif patron_maximo:
                    return f"Máximo local condicionado (dets={[round(d, 4) for d in dets_validos]})"
                else:
                    return f"Punto de silla o indeterminado (dets={[round(d, 4) for d in dets_validos]})"
        
        except Exception as e:
            return f"Error en la clasificación con restricciones: {e}"
    
    def identificar_restricciones_activas_kkt(self, restricciones_desigualdad: List, 
                                            multiplicadores_mu: List, punto: Dict, 
                                            variables: List) -> List[int]:
        """
        Identifica qué restricciones de desigualdad están activas en un punto KKT.
        
        Args:
            restricciones_desigualdad: Lista de restricciones g(x) <= 0
            multiplicadores_mu: Lista de multiplicadores μ
            punto: Diccionario con valores del punto KKT
            variables: Lista de variables originales
        
        Returns:
            Lista de índices de restricciones activas
        """
        vars_originales = self.extraer_variables_originales(punto, variables)
        restricciones_activas = []
        
        print("\nIdentificando restricciones activas:")
        
        for i, restriccion in enumerate(restricciones_desigualdad):
            valor_restriccion = restriccion.subs(vars_originales)
            mu_correspondiente = multiplicadores_mu[i] if i < len(multiplicadores_mu) else None
            valor_mu = punto.get(mu_correspondiente, 0) if mu_correspondiente else 0
            
            try:
                val_rest_num = float(valor_restriccion)
                val_mu_num = float(valor_mu)
                
                # Una restricción está activa si g(x) ≈ 0 o μ > 0
                if abs(val_rest_num) < self.tolerancia or val_mu_num > self.tolerancia:
                    restricciones_activas.append(i)
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (ACTIVA)")
                else:
                    print(f"  Restricción {i+1}: g_{i+1}(x) = {val_rest_num:.6f}, μ_{i+1} = {val_mu_num:.6f} (inactiva)")
            
            except (ValueError, TypeError):
                print(f"  Restricción {i+1}: No se pudo evaluar numéricamente")
        
        return restricciones_activas
    
    def mostrar_punto_detallado(self, punto: Dict, variables: List, funcion_objetivo, 
                                restricciones_igualdad: List = None, 
                                restricciones_desigualdad: List = None,
                                multiplicadores_lambda: List = None,
                                multiplicadores_mu: List = None,
                                clasificacion: str = None) -> None:
        """
        Muestra información detallada de un punto crítico/óptimo.
        
        Args:
            punto: Diccionario con valores del punto
            variables: Lista de variables originales
            funcion_objetivo: Función objetivo simbólica
            restricciones_igualdad: Lista de restricciones de igualdad (opcional)
            restricciones_desigualdad: Lista de restricciones de desigualdad (opcional)
            multiplicadores_lambda: Lista de multiplicadores λ (opcional)
            multiplicadores_mu: Lista de multiplicadores μ (opcional)
            clasificacion: Clasificación del punto (opcional)
        """
        # Extraer variables originales
        vars_originales = self.extraer_variables_originales(punto, variables)
        
        # Mostrar variables originales
        print("  Variables:")
        for var, valor in vars_originales.items():
            print(f"    {var} = {valor}")
        
        # Mostrar multiplicadores si existen
        if multiplicadores_lambda:
            print("  Multiplicadores λ (igualdades):")
            for lam in multiplicadores_lambda:
                if lam in punto:
                    print(f"    {lam} = {punto[lam]}")
        
        if multiplicadores_mu:
            print("  Multiplicadores μ (desigualdades):")
            for mu in multiplicadores_mu:
                if mu in punto:
                    print(f"    {mu} = {punto[mu]}")
        
        # Evaluar función objetivo
        if vars_originales:
            valor_funcion = self.evaluar_funcion_en_punto(funcion_objetivo, vars_originales)
            print(f"  Valor de la función objetivo: f = {valor_funcion}")
        
        # Verificar restricciones de igualdad
        if restricciones_igualdad:
            print("  Verificación de restricciones de igualdad:")
            verificacion = self.verificar_restricciones_igualdad(restricciones_igualdad, vars_originales)
            for i, valor in verificacion:
                if valor is not None:
                    print(f"    h_{i+1} = {valor:.6f} (debe ser ≈ 0)")
                else:
                    print(f"    h_{i+1} = No evaluable")
        
        # Verificar restricciones de desigualdad
        if restricciones_desigualdad:
            print("  Verificación de restricciones de desigualdad:")
            verificacion = self.verificar_restricciones_desigualdad(restricciones_desigualdad, vars_originales)
            for i, valor, activa in verificacion:
                if valor is not None:
                    estado = "ACTIVA" if activa else "inactiva"
                    print(f"    g_{i+1} = {valor:.6f} ≤ 0 ({estado})")
                else:
                    print(f"    g_{i+1} = No evaluable")
        
        # Mostrar clasificación
        if clasificacion:
            print(f"  Clasificación: {clasificacion}")
        else:
            print("  Clasificación: No calculada")
    
    def verificar_condiciones_kkt(self, punto: Dict, restricciones_desigualdad: List, 
                                multiplicadores_mu: List, variables: List) -> bool:
        """
        Verifica si un punto cumple las condiciones KKT.
        
        Args:
            punto: Diccionario con valores del punto
            restricciones_desigualdad: Lista de restricciones g(x) <= 0
            multiplicadores_mu: Lista de multiplicadores μ
            variables: Lista de variables originales
        
        Returns:
            True si cumple las condiciones KKT
        """
        try:
            vars_originales = self.extraer_variables_originales(punto, variables)
            
            # Verificar factibilidad dual: μ ≥ 0
            for mu in multiplicadores_mu:
                if mu in punto:
                    valor_mu = float(punto[mu])
                    if valor_mu < -self.tolerancia:
                        print(f"Violación de factibilidad dual: {mu} = {valor_mu} < 0")
                        return False
            
            # Verificar factibilidad primal: g(x) ≤ 0
            for i, restriccion in enumerate(restricciones_desigualdad):
                valor_restriccion = float(restriccion.subs(vars_originales))
                if valor_restriccion > self.tolerancia:
                    print(f"Violación de factibilidad primal: g_{i+1} = {valor_restriccion} > 0")
                    return False
            
            # Verificar holgura complementaria: μ * g(x) = 0
            for i, (restriccion, mu) in enumerate(zip(restricciones_desigualdad, multiplicadores_mu)):
                if mu in punto:
                    valor_restriccion = float(restriccion.subs(vars_originales))
                    valor_mu = float(punto[mu])
                    producto = valor_mu * valor_restriccion
                    if abs(producto) > self.tolerancia:
                        print(f"Violación de holgura complementaria: μ_{i+1} * g_{i+1} = {producto} ≠ 0")
                        return False
            
            return True
            
        except Exception as e:
            print(f"Error al verificar condiciones KKT: {e}")
            return False