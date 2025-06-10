#!/usr/bin/env python3
"""
Optimizador con Multiplicadores de Lagrange

Este módulo implementa la resolución de problemas de optimización no lineal
con restricciones de igualdad usando el método de los Multiplicadores de Lagrange.
"""

import sympy as sp
from sympy import symbols, diff, solve, pprint, latex
import numpy as np
from typing import List, Dict, Tuple

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
        self.puntos_optimos = []
    
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
    
    def mostrar_puntos_optimos(self):
        """
        Muestra los puntos óptimos y los valores de los multiplicadores
        """
        if not self.puntos_optimos:
            print("\nNo se encontraron puntos óptimos o no se han calculado aún.")
            return
        
        print("\n" + "="*60)
        print("PUNTOS ÓPTIMOS CON RESTRICCIONES (MULTIPLICADORES DE LAGRANGE)")
        print("="*60)
        
        for i, solucion in enumerate(self.puntos_optimos, 1):
            print(f"\nSolución {i}:")
            
            if isinstance(solucion, dict):
                # Separar variables originales y multiplicadores
                vars_originales = {}
                multiplicadores_vals = {}
                
                for var, valor in solucion.items():
                    if var in self.variables:
                        vars_originales[var] = valor
                    elif var in self.multiplicadores:
                        multiplicadores_vals[var] = valor
                
                # Mostrar variables originales
                print("  Variables:")
                for var, valor in vars_originales.items():
                    print(f"    {var} = {valor}")
                
                # Mostrar multiplicadores
                print("  Multiplicadores de Lagrange:")
                for lam, valor in multiplicadores_vals.items():
                    print(f"    {lam} = {valor}")
                
                # Evaluar función objetivo en este punto
                if vars_originales:
                    valor_funcion = self.funcion_objetivo.subs(vars_originales)
                    print(f"  Valor de la función objetivo: f = {valor_funcion}")
                
                # Verificar restricciones
                print("  Verificación de restricciones:")
                for j, restriccion in enumerate(self.restricciones):
                    valor_restriccion = restriccion.subs(vars_originales)
                    print(f"    g_{j+1} = {valor_restriccion} (debe ser ≈ 0)")
    
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
        
        # Paso 8: Mostrar resultados
        self.mostrar_puntos_optimos()

def mostrar_ejemplos_con_restricciones(optimizador):
    """
    Muestra ejemplos predefinidos con restricciones
    """
    ejemplos = [
        {
            "nombre": "Optimización en un círculo",
            "variables": ["x", "y"],
            "funcion": "x + y",
            "restricciones": ["x**2 + y**2 - 1"],
            "descripcion": "Maximizar x+y sujeto a x²+y²=1"
        },
        {
            "nombre": "Mínimo de distancia al origen",
            "variables": ["x", "y"],
            "funcion": "x**2 + y**2",
            "restricciones": ["x + y - 2"],
            "descripcion": "Minimizar x²+y² sujeto a x+y=2"
        },
        {
            "nombre": "Optimización con dos restricciones",
            "variables": ["x", "y", "z"],
            "funcion": "x**2 + y**2 + z**2",
            "restricciones": ["x + y + z - 3", "x - y"],
            "descripcion": "Minimizar x²+y²+z² con x+y+z=3 y x=y"
        },
        {
            "nombre": "Función de utilidad con restricción presupuestaria",
            "variables": ["x", "y"],
            "funcion": "x*y",
            "restricciones": ["2*x + 3*y - 12"],
            "descripcion": "Maximizar xy sujeto a 2x+3y=12"
        }
    ]
    
    print("\nEjemplos CON restricciones disponibles:")
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"{i}. {ejemplo['nombre']}")
        print(f"   Variables: {ejemplo['variables']}")
        print(f"   Función objetivo: {ejemplo['funcion']}")
        print(f"   Restricciones: {ejemplo['restricciones']}")
        print(f"   Descripción: {ejemplo['descripcion']}\n")
    
    try:
        seleccion = int(input("Seleccione un ejemplo (1-4): ")) - 1
        
        if 0 <= seleccion < len(ejemplos):
            ejemplo = ejemplos[seleccion]
            print(f"\nEjecutando: {ejemplo['nombre']}")
            optimizador.analisis_completo_con_restricciones(
                ejemplo['variables'], 
                ejemplo['funcion'], 
                ejemplo['restricciones']
            )
        else:
            print("Selección no válida.")
            
    except ValueError:
        print("Por favor, ingrese un número válido.")

def analisis_con_restricciones_interactivo(optimizador):
    """
    Realiza un análisis con restricciones de forma interactiva
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
        print("  - x**2 + 2*y**2 + 3*z**2")
        
        expresion_objetivo = input("\nFunción objetivo: ").strip()
        
        # Solicitar restricciones
        restricciones = []
        print("\nIngrese las restricciones de igualdad (g(x,y,...) = 0).")
        print("Ingrese solo la parte izquierda de la ecuación (se asume = 0).")
        print("Ejemplos:")
        print("  - x + y - 1  (para x + y = 1)")
        print("  - x**2 + y**2 - 4  (para x² + y² = 4)")
        
        while True:
            restriccion = input(f"\nRestricción {len(restricciones)+1} (o 'fin' para terminar): ").strip()
            if restriccion.lower() == 'fin':
                break
            if restriccion:
                restricciones.append(restriccion)
        
        if not restricciones:
            print("Error: Debe definir al menos una restricción.")
            return
        
        # Realizar análisis
        optimizador.analisis_completo_con_restricciones(nombres_variables, expresion_objetivo, restricciones)
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")