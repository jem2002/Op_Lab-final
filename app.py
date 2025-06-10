#!/usr/bin/env python3
"""
Programa Principal de Optimización No Lineal

Este programa resuelve problemas de optimización no lineal usando tres métodos:
1. Sin restricciones (Cálculo Diferencial)
2. Con restricciones de igualdad (Multiplicadores de Lagrange)
3. Con restricciones de desigualdad (Condiciones KKT)

Autor: Sistema de Optimización
Versión: 3.0
"""

import sys
import os

# Importar los módulos de optimización
from optimizador_sin_restricciones import OptimizadorNoLineal, analisis_completo_interactivo, mostrar_ejemplos_sin_restricciones
from optimizador_lagrange import OptimizadorConRestricciones, analisis_con_restricciones_interactivo, mostrar_ejemplos_con_restricciones
from optimizador_kkt import OptimizadorKKT, analisis_kkt_interactivo, mostrar_ejemplos_kkt

def mostrar_bienvenida():
    """
    Muestra el mensaje de bienvenida del programa
    """
    print("\n" + "="*80)
    print("    PROGRAMA DE OPTIMIZACIÓN NO LINEAL - VERSIÓN COMPLETA")
    print("="*80)
    print("\nEste programa resuelve problemas de optimización no lineal usando:")
    print("\n1. 📊 OPTIMIZACIÓN SIN RESTRICCIONES")
    print("   • Método: Cálculo Diferencial")
    print("   • Encuentra puntos críticos mediante ∇f = 0")
    print("   • Clasifica puntos usando la matriz Hessiana")
    
    print("\n2. 🔗 OPTIMIZACIÓN CON RESTRICCIONES DE IGUALDAD")
    print("   • Método: Multiplicadores de Lagrange")
    print("   • Resuelve problemas con restricciones g(x) = 0")
    print("   • Construye la Lagrangiana L(x,λ) = f(x) - Σ(λᵢ·gᵢ(x))")
    
    print("\n3. ⚖️  OPTIMIZACIÓN CON RESTRICCIONES DE DESIGUALDAD")
    print("   • Método: Condiciones de Karush-Kuhn-Tucker (KKT)")
    print("   • Maneja restricciones g(x) ≤ 0 y h(x) = 0")
    print("   • Verifica condiciones de optimalidad KKT")
    
    print("\n" + "="*80)

def mostrar_menu_principal():
    """
    Muestra el menú principal de opciones
    """
    print("\n🔧 MENÚ PRINCIPAL:")
    print("\n1. Optimización SIN restricciones (Cálculo Diferencial)")
    print("2. Optimización CON restricciones de IGUALDAD (Multiplicadores de Lagrange)")
    print("3. Optimización CON restricciones de DESIGUALDAD (Condiciones KKT)")
    print("4. Ver ejemplos predefinidos")
    print("5. Salir")
    print("\n" + "-"*50)

def mostrar_menu_ejemplos():
    """
    Muestra el menú de ejemplos predefinidos
    """
    print("\n📚 EJEMPLOS PREDEFINIDOS:")
    print("\n1. Ejemplos SIN restricciones")
    print("2. Ejemplos CON restricciones de igualdad (Lagrange)")
    print("3. Ejemplos CON restricciones de desigualdad (KKT)")
    print("4. Volver al menú principal")
    print("\n" + "-"*40)

def ejecutar_optimizacion_sin_restricciones():
    """
    Ejecuta la optimización sin restricciones
    """
    print("\n" + "="*60)
    print("OPTIMIZACIÓN SIN RESTRICCIONES - CÁLCULO DIFERENCIAL")
    print("="*60)
    
    optimizador = OptimizadorNoLineal()
    analisis_completo_interactivo(optimizador)

def ejecutar_optimizacion_con_restricciones_igualdad():
    """
    Ejecuta la optimización con restricciones de igualdad
    """
    print("\n" + "="*60)
    print("OPTIMIZACIÓN CON RESTRICCIONES DE IGUALDAD - MULTIPLICADORES DE LAGRANGE")
    print("="*60)
    
    optimizador = OptimizadorConRestricciones()
    analisis_con_restricciones_interactivo(optimizador)

def ejecutar_optimizacion_kkt():
    """
    Ejecuta la optimización con condiciones KKT
    """
    print("\n" + "="*60)
    print("OPTIMIZACIÓN CON RESTRICCIONES DE DESIGUALDAD - CONDICIONES KKT")
    print("="*60)
    
    optimizador = OptimizadorKKT()
    analisis_kkt_interactivo(optimizador)

def ejecutar_ejemplos():
    """
    Ejecuta los ejemplos predefinidos según la selección del usuario
    """
    while True:
        mostrar_menu_ejemplos()
        
        try:
            opcion = input("\nSeleccione una opción (1-4): ").strip()
            
            if opcion == '1':
                print("\n" + "="*50)
                print("EJEMPLOS SIN RESTRICCIONES")
                print("="*50)
                optimizador = OptimizadorNoLineal()
                mostrar_ejemplos_sin_restricciones(optimizador)
                
            elif opcion == '2':
                print("\n" + "="*50)
                print("EJEMPLOS CON RESTRICCIONES DE IGUALDAD")
                print("="*50)
                optimizador = OptimizadorConRestricciones()
                mostrar_ejemplos_con_restricciones(optimizador)
                
            elif opcion == '3':
                print("\n" + "="*50)
                print("EJEMPLOS CON RESTRICCIONES DE DESIGUALDAD")
                print("="*50)
                optimizador = OptimizadorKKT()
                mostrar_ejemplos_kkt(optimizador)
                
            elif opcion == '4':
                break
                
            else:
                print("❌ Opción no válida. Por favor, seleccione 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Saliendo del menú de ejemplos...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def menu_interactivo():
    """
    Función principal que maneja el menú interactivo
    """
    mostrar_bienvenida()
    
    while True:
        mostrar_menu_principal()
        
        try:
            opcion = input("\nSeleccione una opción (1-5): ").strip()
            
            if opcion == '1':
                ejecutar_optimizacion_sin_restricciones()
                
            elif opcion == '2':
                ejecutar_optimizacion_con_restricciones_igualdad()
                
            elif opcion == '3':
                ejecutar_optimizacion_kkt()
                
            elif opcion == '4':
                ejecutar_ejemplos()
                
            elif opcion == '5':
                print("\n👋 ¡Gracias por usar el programa de optimización!")
                print("🎯 Esperamos que haya sido útil para resolver sus problemas de optimización.")
                break
                
            else:
                print("❌ Opción no válida. Por favor, seleccione 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            print("🔄 Continuando con el programa...")

def verificar_dependencias():
    """
    Verifica que todas las dependencias estén instaladas
    """
    try:
        import sympy
        import numpy
        print("✅ Todas las dependencias están instaladas correctamente.")
        return True
    except ImportError as e:
        print(f"❌ Error: Falta instalar dependencias: {e}")
        print("\n📦 Para instalar las dependencias, ejecute:")
        print("   pip install sympy numpy")
        return False

def main():
    """
    Función principal del programa
    """
    try:
        # Verificar dependencias
        if not verificar_dependencias():
            return
        
        # Verificar que los módulos de optimización existan
        archivos_requeridos = [
            'optimizador_sin_restricciones.py',
            'optimizador_lagrange.py', 
            'optimizador_kkt.py'
        ]
        
        for archivo in archivos_requeridos:
            if not os.path.exists(archivo):
                print(f"❌ Error: No se encuentra el archivo {archivo}")
                print("🔧 Asegúrese de que todos los módulos estén en el mismo directorio.")
                return
        
        print("✅ Todos los módulos de optimización están disponibles.")
        
        # Ejecutar menú interactivo
        menu_interactivo()
        
    except Exception as e:
        print(f"❌ Error crítico en el programa principal: {e}")
        print("🆘 Por favor, verifique la instalación y los archivos del programa.")

if __name__ == "__main__":
    main()