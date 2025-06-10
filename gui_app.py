#!/usr/bin/env python3
"""
Interfaz Gráfica para el Programa de Optimización No Lineal

Esta aplicación proporciona una interfaz gráfica moderna y minimalista
para resolver problemas de optimización no lineal usando tres métodos:
1. Sin restricciones (Cálculo Diferencial)
2. Con restricciones de igualdad (Multiplicadores de Lagrange)
3. Con restricciones de desigualdad (Condiciones KKT)

Autor: Sistema de Optimización
Versión: 4.0 - GUI
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Importar los módulos de optimización
from optimizador_sin_restricciones import OptimizadorNoLineal
from optimizador_lagrange import OptimizadorConRestricciones
from optimizador_kkt import OptimizadorKKT

# Configurar el tema de CustomTkinter
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class OptimizationGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🎯 Optimización No Lineal - Interfaz Gráfica")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Inicializar optimizadores
        self.opt_sin_restricciones = OptimizadorNoLineal()
        self.opt_lagrange = OptimizadorConRestricciones()
        self.opt_kkt = OptimizadorKKT()
        
        # Variables para almacenar restricciones
        self.restricciones_igualdad = []
        self.restricciones_desigualdad = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Frame principal
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título principal
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="🎯 OPTIMIZACIÓN NO LINEAL",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=(20, 10))
        
        subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Resuelve problemas de optimización usando Cálculo Diferencial, Lagrange y KKT",
            font=ctk.CTkFont(size=14)
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Frame para el contenido principal
        content_frame = ctk.CTkFrame(self.main_frame)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Configurar grid
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Panel izquierdo - Controles
        self.setup_left_panel(content_frame)
        
        # Panel derecho - Resultados
        self.setup_right_panel(content_frame)
        
    def setup_left_panel(self, parent):
        """Configura el panel izquierdo con controles"""
        left_frame = ctk.CTkFrame(parent)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Crear un frame scrollable para el contenido
        self.scrollable_frame = ctk.CTkScrollableFrame(
            left_frame,
            width=350,
            height=600
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título del panel
        ctk.CTkLabel(
            self.scrollable_frame,
            text="📊 CONFIGURACIÓN",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 15))
        
        # Método de optimización
        ctk.CTkLabel(
            self.scrollable_frame,
            text="Método de Optimización:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5), anchor="w", padx=20)
        
        self.method_var = ctk.StringVar(value="sin_restricciones")
        methods_frame = ctk.CTkFrame(self.scrollable_frame)
        methods_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkRadioButton(
            methods_frame,
            text="🔢 Sin Restricciones (Cálculo Diferencial)",
            variable=self.method_var,
            value="sin_restricciones",
            command=self.on_method_change
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkRadioButton(
            methods_frame,
            text="🔗 Con Restricciones de Igualdad (Lagrange)",
            variable=self.method_var,
            value="lagrange",
            command=self.on_method_change
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkRadioButton(
            methods_frame,
            text="⚖️ Con Restricciones de Desigualdad (KKT)",
            variable=self.method_var,
            value="kkt",
            command=self.on_method_change
        ).pack(anchor="w", padx=10, pady=5)
        
        # Variables
        ctk.CTkLabel(
            self.scrollable_frame,
            text="Variables (separadas por comas):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 5), anchor="w", padx=20)
        
        self.variables_entry = ctk.CTkEntry(
            self.scrollable_frame,
            placeholder_text="x, y, z",
            height=35
        )
        self.variables_entry.pack(fill="x", padx=20, pady=5)
        
        # Función objetivo
        ctk.CTkLabel(
            self.scrollable_frame,
            text="Función Objetivo:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 5), anchor="w", padx=20)
        
        self.objetivo_entry = ctk.CTkEntry(
            self.scrollable_frame,
            placeholder_text="x**2 + y**2",
            height=35
        )
        self.objetivo_entry.pack(fill="x", padx=20, pady=5)
        
        # Frame para restricciones
        self.restrictions_frame = ctk.CTkFrame(self.scrollable_frame)
        self.restrictions_frame.pack(fill="x", padx=20, pady=15)
        
        self.setup_restrictions_ui()
        
        # Botones de acción
        buttons_frame = ctk.CTkFrame(self.scrollable_frame)
        buttons_frame.pack(fill="x", padx=20, pady=15)
        
        ctk.CTkButton(
            buttons_frame,
            text="🚀 Resolver",
            command=self.solve_optimization,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="📋 Cargar Ejemplo",
            command=self.load_example,
            height=35
        ).pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="🗑️ Limpiar",
            command=self.clear_all,
            height=35
        ).pack(fill="x", padx=10, pady=5)
        
    def setup_restrictions_ui(self):
        """Configura la interfaz para restricciones"""
        # Limpiar frame
        for widget in self.restrictions_frame.winfo_children():
            widget.destroy()
        
        method = self.method_var.get()
        
        if method == "sin_restricciones":
            ctk.CTkLabel(
                self.restrictions_frame,
                text="ℹ️ Sin restricciones necesarias",
                font=ctk.CTkFont(size=12)
            ).pack(pady=10)
            
        elif method == "lagrange":
            ctk.CTkLabel(
                self.restrictions_frame,
                text="Restricciones de Igualdad (g(x) = 0):",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(pady=(10, 5), anchor="w", padx=10)
            
            self.igualdad_entry = ctk.CTkEntry(
                self.restrictions_frame,
                placeholder_text="x + y - 1",
                height=35
            )
            self.igualdad_entry.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkButton(
                self.restrictions_frame,
                text="➕ Agregar Restricción",
                command=self.add_equality_constraint,
                height=30
            ).pack(pady=5)
            
            # Lista de restricciones
            self.igualdad_listbox = tk.Listbox(
                self.restrictions_frame,
                height=3,
                bg="#2b2b2b",
                fg="white",
                selectbackground="#1f538d"
            )
            self.igualdad_listbox.pack(fill="x", padx=10, pady=5)
            
        elif method == "kkt":
            # Restricciones de igualdad
            ctk.CTkLabel(
                self.restrictions_frame,
                text="Restricciones de Igualdad (h(x) = 0):",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(pady=(10, 5), anchor="w", padx=10)
            
            self.kkt_igualdad_entry = ctk.CTkEntry(
                self.restrictions_frame,
                placeholder_text="x + y - 2",
                height=30
            )
            self.kkt_igualdad_entry.pack(fill="x", padx=10, pady=2)
            
            ctk.CTkButton(
                self.restrictions_frame,
                text="➕ Agregar Igualdad",
                command=self.add_kkt_equality_constraint,
                height=25
            ).pack(pady=2)
            
            # Restricciones de desigualdad
            ctk.CTkLabel(
                self.restrictions_frame,
                text="Restricciones de Desigualdad (g(x) ≤ 0):",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(pady=(10, 5), anchor="w", padx=10)
            
            self.kkt_desigualdad_entry = ctk.CTkEntry(
                self.restrictions_frame,
                placeholder_text="-x, -y",
                height=30
            )
            self.kkt_desigualdad_entry.pack(fill="x", padx=10, pady=2)
            
            ctk.CTkButton(
                self.restrictions_frame,
                text="➕ Agregar Desigualdad",
                command=self.add_kkt_inequality_constraint,
                height=25
            ).pack(pady=2)
            
            # Listas de restricciones
            self.kkt_listbox = tk.Listbox(
                self.restrictions_frame,
                height=4,
                bg="#2b2b2b",
                fg="white",
                selectbackground="#1f538d"
            )
            self.kkt_listbox.pack(fill="x", padx=10, pady=5)
    
    def setup_right_panel(self, parent):
        """Configura el panel derecho para mostrar resultados"""
        right_frame = ctk.CTkFrame(parent)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Título del panel
        ctk.CTkLabel(
            right_frame,
            text="📈 RESULTADOS",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 15))
        
        # Área de texto para resultados
        self.results_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            bg="#2b2b2b",
            fg="white",
            insertbackground="white",
            font=("Consolas", 11),
            state=tk.DISABLED
        )
        self.results_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Barra de progreso
        self.progress_bar = ctk.CTkProgressBar(right_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 20))
        self.progress_bar.set(0)
        
    def on_method_change(self):
        """Maneja el cambio de método de optimización"""
        self.setup_restrictions_ui()
        self.clear_restrictions()
        
    def add_equality_constraint(self):
        """Agrega una restricción de igualdad"""
        constraint = self.igualdad_entry.get().strip()
        if constraint:
            self.restricciones_igualdad.append(constraint)
            self.igualdad_listbox.insert(tk.END, f"g_{len(self.restricciones_igualdad)}: {constraint} = 0")
            self.igualdad_entry.delete(0, tk.END)
        
    def add_kkt_equality_constraint(self):
        """Agrega una restricción de igualdad para KKT"""
        constraint = self.kkt_igualdad_entry.get().strip()
        if constraint:
            self.restricciones_igualdad.append(constraint)
            self.kkt_listbox.insert(tk.END, f"h_{len(self.restricciones_igualdad)}: {constraint} = 0")
            self.kkt_igualdad_entry.delete(0, tk.END)
            
    def add_kkt_inequality_constraint(self):
        """Agrega una restricción de desigualdad para KKT"""
        constraint = self.kkt_desigualdad_entry.get().strip()
        if constraint:
            self.restricciones_desigualdad.append(constraint)
            self.kkt_listbox.insert(tk.END, f"g_{len(self.restricciones_desigualdad)}: {constraint} ≤ 0")
            self.kkt_desigualdad_entry.delete(0, tk.END)
    
    def clear_restrictions(self):
        """Limpia todas las restricciones"""
        self.restricciones_igualdad.clear()
        self.restricciones_desigualdad.clear()
        
    def clear_all(self):
        """Limpia todos los campos"""
        self.variables_entry.delete(0, tk.END)
        self.objetivo_entry.delete(0, tk.END)
        self.clear_restrictions()
        self.setup_restrictions_ui()
        self.clear_results()
        
    def clear_results(self):
        """Limpia el área de resultados"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.progress_bar.set(0)
        
    def append_result(self, text):
        """Agrega texto al área de resultados"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
    def load_example(self):
        """Carga un ejemplo predefinido"""
        method = self.method_var.get()
        
        if method == "sin_restricciones":
            self.variables_entry.delete(0, tk.END)
            self.variables_entry.insert(0, "x, y")
            self.objetivo_entry.delete(0, tk.END)
            self.objetivo_entry.insert(0, "x**2 + y**2 - 4*x - 6*y")
            
        elif method == "lagrange":
            self.variables_entry.delete(0, tk.END)
            self.variables_entry.insert(0, "x, y")
            self.objetivo_entry.delete(0, tk.END)
            self.objetivo_entry.insert(0, "x**2 + y**2")
            self.clear_restrictions()
            self.restricciones_igualdad.append("x + y - 1")
            self.setup_restrictions_ui()
            self.igualdad_listbox.insert(tk.END, "g_1: x + y - 1 = 0")
            
        elif method == "kkt":
            self.variables_entry.delete(0, tk.END)
            self.variables_entry.insert(0, "x, y")
            self.objetivo_entry.delete(0, tk.END)
            self.objetivo_entry.insert(0, "x**2 + y**2")
            self.clear_restrictions()
            self.restricciones_desigualdad.append("x + y - 1")
            self.setup_restrictions_ui()
            self.kkt_listbox.insert(tk.END, "g_1: x + y - 1 ≤ 0")
    
    def solve_optimization(self):
        """Resuelve el problema de optimización"""
        # Validar entradas
        variables_text = self.variables_entry.get().strip()
        objetivo_text = self.objetivo_entry.get().strip()
        
        if not variables_text or not objetivo_text:
            messagebox.showerror("Error", "Por favor, complete las variables y la función objetivo.")
            return
        
        # Parsear variables
        try:
            variables = [var.strip() for var in variables_text.split(',')]
        except:
            messagebox.showerror("Error", "Formato de variables inválido. Use: x, y, z")
            return
        
        method = self.method_var.get()
        
        # Validar restricciones según el método
        if method == "lagrange" and not self.restricciones_igualdad:
            messagebox.showerror("Error", "El método de Lagrange requiere al menos una restricción de igualdad.")
            return
        elif method == "kkt" and not self.restricciones_igualdad and not self.restricciones_desigualdad:
            messagebox.showerror("Error", "El método KKT requiere al menos una restricción.")
            return
        
        # Limpiar resultados y mostrar progreso
        self.clear_results()
        self.progress_bar.set(0.1)
        
        # Ejecutar optimización en un hilo separado
        thread = threading.Thread(target=self._solve_optimization_thread, 
                                args=(variables, objetivo_text, method))
        thread.daemon = True
        thread.start()
        
    def _solve_optimization_thread(self, variables, objetivo, method):
        """Ejecuta la optimización en un hilo separado"""
        try:
            # Capturar la salida
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                self.progress_bar.set(0.3)
                
                if method == "sin_restricciones":
                    self.opt_sin_restricciones.analisis_completo(variables, objetivo)
                elif method == "lagrange":
                    self.opt_lagrange.analisis_completo_con_restricciones(
                        variables, objetivo, self.restricciones_igualdad
                    )
                elif method == "kkt":
                    self.opt_kkt.analisis_completo_kkt(
                        variables, objetivo, self.restricciones_igualdad, self.restricciones_desigualdad
                    )
                
                self.progress_bar.set(0.9)
            
            # Mostrar resultados
            output = output_buffer.getvalue()
            if output:
                self.root.after(0, self.append_result, output)
            else:
                self.root.after(0, self.append_result, "✅ Optimización completada sin salida.")
            
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            
        except Exception as e:
            error_msg = f"❌ Error durante la optimización: {str(e)}"
            self.root.after(0, self.append_result, error_msg)
            self.root.after(0, lambda: self.progress_bar.set(0))
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()

def main():
    """Función principal"""
    try:
        app = OptimizationGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("Error Fatal", f"Error al inicializar la aplicación: {e}")

if __name__ == "__main__":
    main()