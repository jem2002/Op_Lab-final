from typing import List, Dict

class FormateadorDidactico:
    """
    Clase para transformar la salida de datos crudos de optimización
    en explicaciones paso a paso, claras y didácticas para estudiantes universitarios.
    
    Maneja tres tipos de problemas:
    1. Optimización sin restricciones
    2. Optimización con restricciones de igualdad (Lagrange)
    3. Optimización con restricciones de desigualdad (KKT)
    """
    
    def __init__(self):
        self.tolerancia = 1e-6
    
    def formatear_expresion(self, expr) -> str:
        """
        Formatea una expresión simbólica para mostrarla de manera clara.
        """
        if expr is None:
            return "No definida"
        try:
            return str(expr)
        except:
            return "Error al formatear"
    
    def formatear_matriz(self, matriz, nombre: str = "Matriz") -> str:
        """
        Formatea una matriz para mostrarla de manera clara.
        """
        if matriz is None:
            return f"{nombre}: No definida"
        
        try:
            filas = matriz.rows
            cols = matriz.cols
            resultado = f"{nombre} ({filas}×{cols}):\n"
            resultado += "\n".join(["  [" + ", ".join([str(matriz[i, j]) for j in range(cols)]) + "]" for i in range(filas)])
            return resultado
        except:
            return f"{nombre}: Error al formatear"
    
    def formatear_punto(self, punto: Dict, variables: List) -> str:
        """
        Formatea un punto para mostrarlo de manera clara.
        """
        if not punto:
            return "Punto no definido"
        
        elementos = []
        for var in variables:
            if var in punto:
                valor = punto[var]
                if isinstance(valor, (int, float)):
                    elementos.append(f"{var} = {valor:.6f}")
                else:
                    elementos.append(f"{var} = {valor}")
        
        return f"({', '.join(elementos)})"
    
    def explicar_sin_restricciones(self, optimizador) -> str:
        """
        Genera explicación didáctica para problemas sin restricciones.
        """
        explicacion = []
        
        # Título del problema
        explicacion.append("🎯 **Problema:** Optimización de la función sin restricciones")
        if optimizador.funcion_objetivo:
            explicacion.append(f"f({', '.join(str(v) for v in optimizador.variables)}) = {self.formatear_expresion(optimizador.funcion_objetivo)}")
        explicacion.append("")
        
        # Paso 1: Calcular el Gradiente
        explicacion.append("📝 **Paso 1: Calcular el Gradiente (∇f)**")
        explicacion.append("El gradiente apunta en la dirección de máximo crecimiento de la función. Los puntos óptimos se encuentran donde el gradiente es cero, es decir, donde ∇f = 0.")
        explicacion.append("")
        
        if optimizador.gradiente:
            explicacion.append("Calculamos las derivadas parciales:")
            for i, var in enumerate(optimizador.variables):
                if i < len(optimizador.gradiente):
                    explicacion.append(f"∂f/∂{var} = {self.formatear_expresion(optimizador.gradiente[i])}")
            
            explicacion.append("")
            explicacion.append("Por lo tanto, el gradiente es:")
            gradiente_str = "∇f = [" + ", ".join([self.formatear_expresion(g) for g in optimizador.gradiente]) + "]"
            explicacion.append(gradiente_str)
        explicacion.append("")
        
        # Paso 2: Encontrar Puntos Críticos
        explicacion.append("⚙️ **Paso 2: Encontrar Puntos Críticos (∇f = 0)**")
        explicacion.append("Para encontrar los puntos críticos, resolvemos el sistema de ecuaciones que resulta de igualar cada componente del gradiente a cero.")
        explicacion.append("")
        
        if optimizador.gradiente:
            explicacion.append("Sistema de ecuaciones:")
            for i, grad_comp in enumerate(optimizador.gradiente):
                explicacion.append(f"  {self.formatear_expresion(grad_comp)} = 0")
        
        if optimizador.puntos_criticos:
            explicacion.append("")
            explicacion.append("**Puntos críticos encontrados:**")
            for i, punto in enumerate(optimizador.puntos_criticos):
                punto_str = self.formatear_punto(punto, optimizador.variables)
                explicacion.append(f"  P_{i+1} = {punto_str}")
        explicacion.append("")
        
        # Paso 3: Clasificar con la Matriz Hessiana
        explicacion.append("🔍 **Paso 3: Clasificar los Puntos con la Matriz Hessiana (H)**")
        explicacion.append("La matriz Hessiana contiene las segundas derivadas parciales y nos ayuda a determinar la naturaleza de cada punto crítico.")
        explicacion.append("")
        
        if optimizador.hessiana is not None:
            explicacion.append("La matriz Hessiana es:")
            explicacion.append(self.formatear_matriz(optimizador.hessiana, "H"))
            explicacion.append("")
            
            # Clasificación de cada punto
            if optimizador.puntos_criticos and optimizador.clasificacion_puntos:
                explicacion.append("**Clasificación de los puntos críticos:**")
                for i, (punto, clasificacion) in enumerate(zip(optimizador.puntos_criticos, optimizador.clasificacion_puntos)):
                    punto_str = self.formatear_punto(punto, optimizador.variables)
                    explicacion.append(f"")
                    explicacion.append(f"En P_{i+1} = {punto_str}:")
                    
                    # Evaluar Hessiana en el punto
                    try:
                        hessiana_evaluada = optimizador.hessiana.subs(punto)
                        explicacion.append(f"H(P_{i+1}) = {self.formatear_matriz(hessiana_evaluada, '')}")
                    except:
                        explicacion.append(f"H(P_{i+1}) = [Error al evaluar]")
                    
                    # Interpretación de la clasificación
                    if "mínimo" in clasificacion.lower():
                        explicacion.append(f"La matriz es **definida positiva**, por lo tanto P_{i+1} es un **mínimo local**.")
                    elif "máximo" in clasificacion.lower():
                        explicacion.append(f"La matriz es **definida negativa**, por lo tanto P_{i+1} es un **máximo local**.")
                    elif "silla" in clasificacion.lower():
                        explicacion.append(f"La matriz es **indefinida**, por lo tanto P_{i+1} es un **punto de silla**.")
                    else:
                        explicacion.append(f"Clasificación: {clasificacion}")
        
        explicacion.append("")
        
        # Solución Final
        explicacion.append("✅ **Solución Final**")
        if optimizador.puntos_criticos and optimizador.clasificacion_puntos:
            explicacion.append("**Resumen de los hallazgos:**")
            for i, (punto, clasificacion) in enumerate(zip(optimizador.puntos_criticos, optimizador.clasificacion_puntos)):
                punto_str = self.formatear_punto(punto, optimizador.variables)
                if optimizador.funcion_objetivo:
                    try:
                        valor_funcion = optimizador.funcion_objetivo.subs(punto)
                        explicacion.append(f"• P_{i+1} = {punto_str} es un **{clasificacion}** con f(P_{i+1}) = {valor_funcion}")
                    except:
                        explicacion.append(f"• P_{i+1} = {punto_str} es un **{clasificacion}**")
                else:
                    explicacion.append(f"• P_{i+1} = {punto_str} es un **{clasificacion}**")
        else:
            explicacion.append("No se encontraron puntos críticos o no se pudieron clasificar.")
        
        return "\n".join(explicacion)
    
    def explicar_lagrange(self, optimizador) -> str:
        """
        Genera explicación didáctica para problemas con restricciones de igualdad (Lagrange).
        """
        explicacion = []
        
        # Título del problema
        explicacion.append("🎯 **Problema:** Optimización con restricciones de igualdad")
        if optimizador.funcion_objetivo:
            explicacion.append(f"Optimizar f({', '.join(str(v) for v in optimizador.variables)}) = {self.formatear_expresion(optimizador.funcion_objetivo)}")
        
        if optimizador.restricciones:
            explicacion.append("Sujeto a las restricciones:")
            for i, restriccion in enumerate(optimizador.restricciones):
                explicacion.append(f"  g_{i+1}({', '.join(str(v) for v in optimizador.variables)}) = {self.formatear_expresion(restriccion)} = 0")
        explicacion.append("")
        
        # Paso 1: Construir la Función Lagrangiana
        explicacion.append("📝 **Paso 1: Construir la Función Lagrangiana (L(x, λ))**")
        explicacion.append("El método de Lagrange introduce multiplicadores (λ) para incorporar las restricciones en la función objetivo.")
        explicacion.append("")
        
        if optimizador.restricciones:
            explicacion.append("La función Lagrangiana se define como:")
            lagrangiana_formula = f"L(x, λ) = f(x) - Σ(λᵢ × gᵢ(x))"
            explicacion.append(lagrangiana_formula)
            explicacion.append("")
        
        if optimizador.lagrangiana:
            explicacion.append("Para nuestro problema específico:")
            vars_str = ', '.join(str(v) for v in optimizador.variables)
            mults_str = ', '.join(str(m) for m in optimizador.multiplicadores) if optimizador.multiplicadores else 'λ'
            explicacion.append(f"L({vars_str}, {mults_str}) = {self.formatear_expresion(optimizador.lagrangiana)}")
        explicacion.append("")
        
        # Paso 2: Condiciones de Primer Orden
        explicacion.append("⚙️ **Paso 2: Condiciones de Primer Orden**")
        explicacion.append("Los candidatos a óptimos se encuentran donde el gradiente de la Lagrangiana es cero.")
        explicacion.append("")
        
        if optimizador.gradiente_lagrangiana:
            explicacion.append("Establecemos el sistema de ecuaciones:")
            
            # Derivadas respecto a las variables originales
            for i, var in enumerate(optimizador.variables):
                if i < len(optimizador.gradiente_lagrangiana):
                    explicacion.append(f"  ∂L/∂{var} = {self.formatear_expresion(optimizador.gradiente_lagrangiana[i])} = 0")
            
            # Derivadas respecto a los multiplicadores (restricciones)
            if optimizador.multiplicadores:
                for i, mult in enumerate(optimizador.multiplicadores):
                    idx = len(optimizador.variables) + i
                    if idx < len(optimizador.gradiente_lagrangiana):
                        explicacion.append(f"  ∂L/∂{mult} = {self.formatear_expresion(optimizador.gradiente_lagrangiana[idx])} = 0")
        explicacion.append("")
        
        # Paso 3: Resolver el Sistema
        explicacion.append("🔍 **Paso 3: Resolver el Sistema**")
        explicacion.append("Resolvemos el sistema de ecuaciones para encontrar los puntos candidatos.")
        explicacion.append("")
        
        if optimizador.puntos_optimos:
            explicacion.append("**Puntos candidatos encontrados:**")
            for i, punto in enumerate(optimizador.puntos_optimos):
                explicacion.append(f"")
                explicacion.append(f"**Solución {i+1}:**")
                
                # Variables originales
                vars_originales = {var: punto.get(var, var) for var in optimizador.variables if var in punto}
                punto_str = self.formatear_punto(vars_originales, optimizador.variables)
                explicacion.append(f"  Variables: {punto_str}")
                
                # Multiplicadores de Lagrange
                if optimizador.multiplicadores:
                    mults_valores = []
                    for mult in optimizador.multiplicadores:
                        if mult in punto:
                            valor = punto[mult]
                            if isinstance(valor, (int, float)):
                                mults_valores.append(f"{mult} = {valor:.6f}")
                            else:
                                mults_valores.append(f"{mult} = {valor}")
                    if mults_valores:
                        explicacion.append(f"  Multiplicadores: {', '.join(mults_valores)}")
                
                # Valor de la función objetivo
                if optimizador.funcion_objetivo:
                    try:
                        valor_funcion = optimizador.funcion_objetivo.subs(vars_originales)
                        explicacion.append(f"  Valor de f: {valor_funcion}")
                    except:
                        explicacion.append(f"  Valor de f: [Error al evaluar]")
        explicacion.append("")
        
        # Solución Final
        explicacion.append("✅ **Solución Final**")
        if optimizador.puntos_optimos:
            explicacion.append("**Resumen de los hallazgos:**")
            for i, punto in enumerate(optimizador.puntos_optimos):
                vars_originales = {var: punto.get(var, var) for var in optimizador.variables if var in punto}
                punto_str = self.formatear_punto(vars_originales, optimizador.variables)
                
                if optimizador.funcion_objetivo:
                    try:
                        valor_funcion = optimizador.funcion_objetivo.subs(vars_originales)
                        explicacion.append(f"• El punto óptimo que satisface las restricciones es x* = {punto_str} con f(x*) = {valor_funcion}")
                    except:
                        explicacion.append(f"• El punto óptimo que satisface las restricciones es x* = {punto_str}")
                
                # Mostrar multiplicadores
                if optimizador.multiplicadores:
                    mults_valores = []
                    for mult in optimizador.multiplicadores:
                        if mult in punto:
                            valor = punto[mult]
                            mults_valores.append(f"{mult}* = {valor}")
                    if mults_valores:
                        explicacion.append(f"  Los multiplicadores de Lagrange son: {', '.join(mults_valores)}")
        else:
            explicacion.append("No se encontraron puntos óptimos.")
        
        return "\n".join(explicacion)
    
    def explicar_kkt(self, optimizador) -> str:
        """
        Genera explicación didáctica para problemas con restricciones de desigualdad (KKT).
        """
        explicacion = []
        
        # Título del problema
        explicacion.append("🎯 **Problema:** Optimización con restricciones de desigualdad")
        if optimizador.funcion_objetivo:
            explicacion.append(f"Optimizar f({', '.join(str(v) for v in optimizador.variables)}) = {self.formatear_expresion(optimizador.funcion_objetivo)}")
        
        if hasattr(optimizador, 'restricciones_desigualdad') and optimizador.restricciones_desigualdad:
            explicacion.append("Sujeto a las restricciones:")
            for i, restriccion in enumerate(optimizador.restricciones_desigualdad):
                explicacion.append(f"  g_{i+1}({', '.join(str(v) for v in optimizador.variables)}) = {self.formatear_expresion(restriccion)} ≤ 0")
        explicacion.append("")
        
        # Paso 1: Plantear las Condiciones KKT
        explicacion.append("📝 **Paso 1: Plantear las Condiciones de Karush-Kuhn-Tucker (KKT)**")
        explicacion.append("KKT generaliza el método de Lagrange para restricciones de desigualdad.")
        explicacion.append("")
        
        if hasattr(optimizador, 'lagrangiana_kkt') and optimizador.lagrangiana_kkt:
            explicacion.append("La Lagrangiana se define como:")
            vars_str = ', '.join(str(v) for v in optimizador.variables)
            mults_str = ', '.join(str(m) for m in optimizador.multiplicadores_mu) if hasattr(optimizador, 'multiplicadores_mu') else 'μ'
            explicacion.append(f"L({vars_str}, {mults_str}) = {self.formatear_expresion(optimizador.lagrangiana_kkt)}")
            explicacion.append("")
        
        explicacion.append("**Las 4 condiciones KKT para este problema son:**")
        explicacion.append("1. **Estacionariedad:** ∇f(x) + μ∇g(x) = 0")
        explicacion.append("2. **Factibilidad Primal:** g(x) ≤ 0")
        explicacion.append("3. **Factibilidad Dual:** μ ≥ 0")
        explicacion.append("4. **Holgura Complementaria:** μ × g(x) = 0")
        explicacion.append("")
        
        # Paso 2: Analizar los Casos de la Holgura Complementaria
        explicacion.append("⚙️ **Paso 2: Analizar los Casos de la Holgura Complementaria**")
        explicacion.append("La condición de holgura complementaria crea dos escenarios posibles:")
        explicacion.append("")
        
        explicacion.append("**Caso 1: Restricción Inactiva (μ = 0)**")
        explicacion.append("Si la restricción no es relevante, tratamos el problema como si no tuviera restricciones.")
        explicacion.append("Resolvemos ∇f(x) = 0 y verificamos si el candidato cumple g(x) ≤ 0.")
        explicacion.append("")
        
        explicacion.append("**Caso 2: Restricción Activa (g(x) = 0)**")
        explicacion.append("Si la solución está justo en la frontera de la restricción.")
        explicacion.append("Resolvemos el sistema ∇f(x) + μ∇g(x) = 0 y g(x) = 0, verificando que μ ≥ 0.")
        explicacion.append("")
        
        # Paso 3: Mostrar candidatos encontrados
        if hasattr(optimizador, 'candidatos_kkt') and optimizador.candidatos_kkt:
            explicacion.append("🔍 **Paso 3: Candidatos Encontrados**")
            
            for i, candidato in enumerate(optimizador.candidatos_kkt):
                explicacion.append(f"")
                explicacion.append(f"**Candidato {i+1}:**")
                
                # Mostrar el punto
                if 'punto' in candidato:
                    punto_str = self.formatear_punto(candidato['punto'], optimizador.variables)
                    explicacion.append(f"  Punto: {punto_str}")
                
                # Mostrar el caso
                if 'caso' in candidato:
                    explicacion.append(f"  Caso: {candidato['caso']}")
                
                # Mostrar si es válido
                if 'valido' in candidato:
                    estado = "Válido" if candidato['valido'] else "No válido"
                    explicacion.append(f"  Estado: {estado}")
                
                # Mostrar valor de la función
                if 'valor_funcion' in candidato:
                    explicacion.append(f"  f(x) = {candidato['valor_funcion']}")
        
        explicacion.append("")
        
        # Solución Final
        explicacion.append("✅ **Solución Final**")
        if hasattr(optimizador, 'solucion_optima_kkt') and optimizador.solucion_optima_kkt:
            punto_str = self.formatear_punto(optimizador.solucion_optima_kkt, optimizador.variables)
            if optimizador.funcion_objetivo:
                try:
                    valor_funcion = optimizador.funcion_objetivo.subs(optimizador.solucion_optima_kkt)
                    explicacion.append(f"La solución óptima es x* = {punto_str} con f(x*) = {valor_funcion}")
                except:
                    explicacion.append(f"La solución óptima es x* = {punto_str}")
            else:
                explicacion.append(f"La solución óptima es x* = {punto_str}")
        else:
            explicacion.append("No se encontró una solución óptima válida.")
        
        return "\n".join(explicacion)
    
    def generar_explicacion_completa(self, optimizador, tipo_problema: str) -> str:
        """
        Genera la explicación completa según el tipo de problema.
        
        Args:
            optimizador: Instancia del optimizador correspondiente
            tipo_problema: 'sin_restricciones', 'lagrange', o 'kkt'
        
        Returns:
            String con la explicación didáctica completa
        """
        try:
            if tipo_problema == 'sin_restricciones':
                return self.explicar_sin_restricciones(optimizador)
            elif tipo_problema == 'lagrange':
                return self.explicar_lagrange(optimizador)
            elif tipo_problema == 'kkt':
                return self.explicar_kkt(optimizador)
            else:
                return f"Tipo de problema '{tipo_problema}' no reconocido."
        except Exception as e:
            return f"Error al generar la explicación: {str(e)}"