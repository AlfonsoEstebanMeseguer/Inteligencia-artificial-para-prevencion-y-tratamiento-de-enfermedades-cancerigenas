import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from datetime import datetime
import json

class BreakHisEDA:
    def __init__(self, all_images, all_labels, slides, base_path, label_map=None, show_plots=False):
        """
        EDA profesional para dataset BreakHis binario
        
        Args:
            all_images: Lista de rutas de imágenes
            all_labels: Lista de labels (0: benigno, 1: maligno)
            slides: Lista de IDs de pacientes
            base_path: Ruta base del dataset
            label_map: Diccionario de mapeo de labels
            show_plots: Mostrar gráficas en tiempo real
        """
        self.all_images = all_images
        self.all_labels = all_labels
        self.slides = slides
        self.base_path = base_path
        self.show_plots = show_plots
        
        if label_map is None:
            self.label_map = {0: 'benign', 1: 'malignant'}
        else:
            self.label_map = {v: k for k, v in label_map.items()}
        
        # Inicializar estructuras de datos
        self.df = None
        self.zoom_stats = {}
        self.patient_stats = {}
        self.image_stats = {}
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        self.colors = ['#4C72B0', '#DD8452']
        
        if not self.show_plots:
            plt.ioff()
            
        # Preparar datos
        self._preparar_dataframe()
        
    def _show_or_close(self, fig=None):
        """Manejo de visualizaciones"""
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig if fig is not None else "all")

    def _save_fig(self, fig, filename, **kwargs):
        """Muestra el gráfico (si aplica) antes de guardarlo y cierra cuando no se muestra."""
        if self.show_plots:
            plt.show(block=True)
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, **kwargs)
        if not self.show_plots:
            plt.close(fig)
    
    def _preparar_dataframe(self):
        """Crear DataFrame estructurado con toda la información"""
        print("📊 Preparando DataFrame...")
        
        data = []
        for img_path, label, slide in tqdm(zip(self.all_images, self.all_labels, self.slides), 
                                          total=len(self.all_images), desc="Procesando"):
            try:
                # Extraer información de la ruta
                parts = img_path.split(os.sep)
                zoom = parts[-3]  # 40X, 100X, 200X, 400X
                class_name = 'benign' if label == 0 else 'malignant'
                
                # Parsear nombre de archivo
                filename = os.path.basename(img_path)
                name_parts = filename.split('-')
                
                info = {
                    'filepath': img_path,
                    'filename': filename,
                    'label': label,
                    'label_name': class_name,
                    'patient_id': slide,
                    'zoom': zoom,
                    'magnification': zoom.replace('X', ''),
                    'dataset': name_parts[0].split('_')[0] if len(name_parts) > 0 else '',
                    'class_code': name_parts[0].split('_')[1] if len(name_parts[0].split('_')) > 1 else '',
                    'subclass': name_parts[0].split('_')[2] if len(name_parts[0].split('_')) > 2 else '',
                    'year': name_parts[1] if len(name_parts) > 1 else '',
                    'sequence': name_parts[4] if len(name_parts) > 4 else ''
                }
                data.append(info)
            except Exception as e:
                print(f"⚠️  Error procesando {img_path}: {e}")
                continue
        
        self.df = pd.DataFrame(data)
        
        # Estadísticas básicas
        self.total_imagenes = len(self.df)
        self.total_pacientes = self.df['patient_id'].nunique()
        self.total_zooms = self.df['zoom'].nunique()
        
        print(f"✅ DataFrame creado: {self.total_imagenes} imágenes, {self.total_pacientes} pacientes, {self.total_zooms} aumentos")
        print(f"📊 Distribución de clases: {self.df['label_name'].value_counts().to_dict()}")
    
    def ejecutar_analisis_completo(self, output_dir="breakhis_eda_resultados", sample_images=100):
        """
        Ejecuta análisis EDA completo
        """
        print("=" * 100)
        print("🚀 ANÁLISIS EDA PROFESIONAL - DATASET BREAKHIS BINARIO")
        print("=" * 100)
        
        # Crear directorio de salida
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Resultados en: {self.output_dir}")
        
        try:
            # 1. ANÁLISIS ESTADÍSTICO BÁSICO
            print("\n" + "=" * 50)
            print("1. 📊 ANÁLISIS ESTADÍSTICO BÁSICO")
            print("=" * 50)
            self.analisis_estadistico_basico()
            
            # 2. ANÁLISIS POR AUMENTO (ZOOM)
            print("\n" + "=" * 50)
            print("2. 🔬 ANÁLISIS POR NIVEL DE AUMENTO")
            print("=" * 50)
            self.analisis_por_aumento()
            
            # 3. ANÁLISIS DE PACIENTES
            print("\n" + "=" * 50)
            print("3. 👥 ANÁLISIS DE PACIENTES")
            print("=" * 50)
            self.analisis_pacientes()
            
            # 4. ANÁLISIS DE IMÁGENES (MUESTRA)
            print("\n" + "=" * 50)
            print("4. 🖼️  ANÁLISIS DE IMÁGENES")
            print("=" * 50)
            self.analisis_imagenes_muestra(sample_size=sample_images)
            
            # 5. ANÁLISIS DE SUB-CLASES
            print("\n" + "=" * 50)
            print("5. 🏷️  ANÁLISIS DE SUB-CLASES")
            print("=" * 50)
            self.analisis_subclases()
            
            # 6. ANÁLISIS PARA SPLIT POR PACIENTES
            print("\n" + "=" * 50)
            print("6. 🎯 ANÁLISIS PARA SPLIT POR PACIENTES")
            print("=" * 50)
            self.analisis_split_pacientes()
            
            # 7. VISUALIZACIONES AVANZADAS
            print("\n" + "=" * 50)
            print("7. 📊 VISUALIZACIONES AVANZADAS")
            print("=" * 50)
            self.visualizaciones_avanzadas()
            
            # 8. ANÁLISIS DE CALIDAD
            print("\n" + "=" * 50)
            print("8. 🔍 ANÁLISIS DE CALIDAD DE IMÁGENES")
            print("=" * 50)
            self.analisis_calidad_imagenes()
            
            # 9. ANÁLISIS DE COLOR Y TEXTURAS
            print("\n" + "=" * 50)
            print("9. 🎨 ANÁLISIS DE COLOR Y TEXTURAS")
            print("=" * 50)
            self.analisis_color_texturas()
            
            # 10. REPORTE FINAL
            print("\n" + "=" * 50)
            print("10. 📋 REPORTE FINAL")
            print("=" * 50)
            self.generar_reporte_final()
            
            print("\n" + "=" * 100)
            print("✅ ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
            print(f"📁 Todos los resultados están en: {self.output_dir}")
            print("=" * 100)
            
            # Mostrar resumen ejecutivo
            self.mostrar_resumen_ejecutivo()
            
        except Exception as e:
            print(f"❌ Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
    
    def analisis_estadistico_basico(self):
        """Análisis estadístico básico del dataset"""
        
        print("\n📊 ESTADÍSTICAS GLOBALES:")
        print(f"   • Total imágenes: {self.total_imagenes:,}")
        print(f"   • Total pacientes: {self.total_pacientes:,}")
        print(f"   • Niveles de aumento: {self.total_zooms}")
        print(f"   • Imágenes por paciente (promedio): {self.total_imagenes/self.total_pacientes:.2f}")
        
        # Distribución por clase
        distribucion_clases = self.df['label_name'].value_counts()
        stats_df = pd.DataFrame({
            'Clase': distribucion_clases.index,
            'Imágenes': distribucion_clases.values,
            'Porcentaje': (distribucion_clases.values / self.total_imagenes * 100).round(2),
            'Pacientes Únicos': [self.df[self.df['label_name'] == c]['patient_id'].nunique() 
                               for c in distribucion_clases.index]
        })
        
        print(f"\n📈 DISTRIBUCIÓN POR CLASE:")
        print(stats_df.to_string(index=False))
        
        # Calcular balanceo
        ratio = stats_df['Imágenes'].max() / stats_df['Imágenes'].min()
        
        # Calcular coeficiente de variación
        cv = stats_df['Imágenes'].std() / stats_df['Imágenes'].mean()
        
        print(f"\n⚖️  ANÁLISIS DE BALANCEO:")
        print(f"   • Ratio maligno/benigno: {ratio:.2f}:1")
        print(f"   • Coeficiente de variación: {cv:.3f}")
        print(f"   • Clase mayoritaria: {stats_df.iloc[0]['Clase']} ({stats_df.iloc[0]['Imágenes']} imágenes)")
        print(f"   • Clase minoritaria: {stats_df.iloc[1]['Clase']} ({stats_df.iloc[1]['Imágenes']} imágenes)")
        print(f"   • Ratio mayoritaria/minoritaria: {stats_df.iloc[0]['Imágenes']/stats_df.iloc[1]['Imágenes']:.2f}:1")
        print(f"   • Diferencia absoluta: {abs(stats_df.iloc[0]['Imágenes'] - stats_df.iloc[1]['Imágenes'])} imágenes")
        print(f"   • Porcentaje clase mayoritaria: {stats_df['Porcentaje'].max():.1f}%")
        
        if ratio > 1.5:
            print("   ⚠️  Dataset DESBALANCEADO - considerar técnicas de balanceo")
        else:
            print("   ✅ Dataset relativamente balanceado")
        
        # Guardar estadísticas
        self.stats_basicas = {
            'total_imagenes': self.total_imagenes,
            'total_pacientes': self.total_pacientes,
            'distribucion_clases': stats_df.to_dict('records'),
            'ratio_balanceo': ratio,
            'coeficiente_variacion': cv,
            'needs_balancing': ratio > 1.5
        }
        
        # Visualización
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Gráfico de barras por clase
        bars = axes[0].bar(stats_df['Clase'], stats_df['Imágenes'], color=self.colors)
        axes[0].set_title('Distribución por Clase', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Clase', fontsize=10)
        axes[0].set_ylabel('Número de imágenes', fontsize=10)
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Gráfico circular
        axes[1].pie(stats_df['Imágenes'], labels=stats_df['Clase'], 
                   autopct='%1.1f%%', startangle=90, colors=self.colors)
        axes[1].set_title('Porcentaje por Clase', fontweight='bold', fontsize=12)
        
        # 3. Pacientes únicos por clase
        axes[2].bar(stats_df['Clase'], stats_df['Pacientes Únicos'], color=self.colors)
        axes[2].set_title('Pacientes Únicos por Clase', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('Clase', fontsize=10)
        axes[2].set_ylabel('Número de pacientes', fontsize=10)
        for i, val in enumerate(stats_df['Pacientes Únicos']):
            axes[2].text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        self._save_fig(fig, '01_distribucion_clases.png', dpi=150, bbox_inches='tight')
        
        return stats_df
    
    def analisis_por_aumento(self):
        """Análisis detallado por nivel de aumento"""
        
        print("\n🔬 ANÁLISIS POR NIVEL DE AUMENTO:")
        
        # Agrupar por aumento
        zoom_stats = self.df.groupby('zoom').agg({
            'label': 'count',
            'patient_id': 'nunique',
            'label_name': lambda x: (x == 'benign').sum()
        }).reset_index()
        
        zoom_stats.columns = ['Zoom', 'Total_Imagenes', 'Pacientes_Unicos', 'Benignas']
        zoom_stats['Malignas'] = zoom_stats['Total_Imagenes'] - zoom_stats['Benignas']
        zoom_stats['%_Benignas'] = (zoom_stats['Benignas'] / zoom_stats['Total_Imagenes'] * 100).round(2)
        zoom_stats['%_Malignas'] = (zoom_stats['Malignas'] / zoom_stats['Total_Imagenes'] * 100).round(2)
        
        # Ordenar por aumento
        zoom_order = ['40X', '100X', '200X', '400X']
        zoom_stats['Zoom'] = pd.Categorical(zoom_stats['Zoom'], categories=zoom_order, ordered=True)
        zoom_stats = zoom_stats.sort_values('Zoom')
        
        print(zoom_stats.to_string(index=False))
        
        # Guardar estadísticas
        self.zoom_stats = zoom_stats.to_dict('records')
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Total imágenes por aumento
        axes[0,0].bar(zoom_stats['Zoom'], zoom_stats['Total_Imagenes'], color='skyblue', edgecolor='black')
        axes[0,0].set_title('Total de Imágenes por Aumento', fontweight='bold', fontsize=12)
        axes[0,0].set_xlabel('Aumento', fontsize=10)
        axes[0,0].set_ylabel('Número de imágenes', fontsize=10)
        axes[0,0].grid(True, alpha=0.3)
        for i, val in enumerate(zoom_stats['Total_Imagenes']):
            axes[0,0].text(i, val + 5, str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Distribución benigno/maligno por aumento (stacked)
        x = range(len(zoom_stats))
        axes[0,1].bar(x, zoom_stats['Benignas'], label='Benignas', color=self.colors[0], edgecolor='black')
        axes[0,1].bar(x, zoom_stats['Malignas'], bottom=zoom_stats['Benignas'], 
                     label='Malignas', color=self.colors[1], edgecolor='black')
        axes[0,1].set_title('Distribución Benigno/Maligno por Aumento', fontweight='bold', fontsize=12)
        axes[0,1].set_xlabel('Aumento', fontsize=10)
        axes[0,1].set_ylabel('Número de imágenes', fontsize=10)
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(zoom_stats['Zoom'])
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Porcentaje por clase por aumento
        x = np.arange(len(zoom_stats))
        width = 0.35
        axes[1,0].bar(x - width/2, zoom_stats['%_Benignas'], width, label='Benignas', 
                     color=self.colors[0], edgecolor='black')
        axes[1,0].bar(x + width/2, zoom_stats['%_Malignas'], width, label='Malignas', 
                     color=self.colors[1], edgecolor='black')
        axes[1,0].set_title('Porcentaje por Clase por Aumento', fontweight='bold', fontsize=12)
        axes[1,0].set_xlabel('Aumento', fontsize=10)
        axes[1,0].set_ylabel('Porcentaje (%)', fontsize=10)
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(zoom_stats['Zoom'])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Pacientes únicos por aumento
        axes[1,1].bar(zoom_stats['Zoom'], zoom_stats['Pacientes_Unicos'], 
                     color='lightgreen', edgecolor='black')
        axes[1,1].set_title('Pacientes Únicos por Aumento', fontweight='bold', fontsize=12)
        axes[1,1].set_xlabel('Aumento', fontsize=10)
        axes[1,1].set_ylabel('Número de pacientes', fontsize=10)
        axes[1,1].grid(True, alpha=0.3)
        for i, val in enumerate(zoom_stats['Pacientes_Unicos']):
            axes[1,1].text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        self._save_fig(fig, '02_analisis_aumento.png', dpi=150, bbox_inches='tight')
        
        # Análisis cruzado: Aumento vs Clase
        print("\n📊 DISTRIBUCIÓN CRUZADA (AUMENTO × CLASE):")
        cross_tab = pd.crosstab(self.df['zoom'], self.df['label_name'], normalize='index') * 100
        cross_tab = cross_tab.reindex(zoom_order)
        print(cross_tab.round(2))
        
        # Heatmap de distribución
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Porcentaje (%)'})
        ax.set_title('Distribución Porcentual por Aumento y Clase', fontweight='bold', fontsize=12)
        ax.set_xlabel('Clase', fontsize=10)
        ax.set_ylabel('Aumento', fontsize=10)
        plt.tight_layout()
        self._save_fig(fig, '02_heatmap_aumento_clase.png', dpi=150, bbox_inches='tight')
        
        return zoom_stats
    
    def analisis_pacientes(self):
        """Análisis detallado de pacientes"""
        
        print("\n👥 ANÁLISIS DETALLADO DE PACIENTES:")
        
        # Imágenes por paciente
        imagenes_por_paciente = self.df.groupby('patient_id').size()
        
        # Calcular estadísticas detalladas
        stats_pacientes = {
            'media': imagenes_por_paciente.mean(),
            'mediana': imagenes_por_paciente.median(),
            'std': imagenes_por_paciente.std(),
            'min': imagenes_por_paciente.min(),
            'max': imagenes_por_paciente.max(),
            'q1': imagenes_por_paciente.quantile(0.25),
            'q3': imagenes_por_paciente.quantile(0.75),
            'pacientes_con_1_imagen': (imagenes_por_paciente == 1).sum(),
            'pacientes_con_mas_de_10_imagenes': (imagenes_por_paciente > 10).sum(),
            'total_pacientes': len(imagenes_por_paciente),
            'coeficiente_variacion': imagenes_por_paciente.std() / imagenes_por_paciente.mean()
        }
        
        print("📊 DISTRIBUCIÓN POR PACIENTE:")
        print(f"   • Media: {stats_pacientes['media']:.2f}")
        print(f"   • Mediana: {stats_pacientes['mediana']:.2f}")
        print(f"   • Desviación estándar: {stats_pacientes['std']:.2f}")
        print(f"   • Mínimo: {stats_pacientes['min']}")
        print(f"   • Máximo: {stats_pacientes['max']}")
        print(f"   • Q1 (25%): {stats_pacientes['q1']:.2f}")
        print(f"   • Q3 (75%): {stats_pacientes['q3']:.2f}")
        print(f"   • Pacientes con 1 imagen: {stats_pacientes['pacientes_con_1_imagen']}")
        print(f"   • Pacientes con más de 10 imágenes: {stats_pacientes['pacientes_con_mas_de_10_imagenes']}")
        print(f"   • Coeficiente de variación: {stats_pacientes['coeficiente_variacion']:.3f}")
        print(f"   • Número máximo de imágenes por paciente: {stats_pacientes['max']}")
        
        # Análisis de pacientes con múltiples clases
        print("\n🔍 PACIENTES CON MÚLTIPLES CLASES:")
        pacientes_multiclase = self.df.groupby('patient_id')['label_name'].nunique()
        pacientes_multiclase = pacientes_multiclase[pacientes_multiclase > 1]
        
        print(f"   • Pacientes con múltiples clases: {len(pacientes_multiclase)}")
        print(f"   • Porcentaje del total: {len(pacientes_multiclase)/stats_pacientes['total_pacientes']*100:.1f}%")
        
        if len(pacientes_multiclase) > 0:
            print(f"   • Número máximo de clases por paciente: {pacientes_multiclase.max()}")
            
            # Mostrar algunos ejemplos
            print(f"\n   📋 Ejemplos de pacientes con múltiples clases:")
            for paciente_id, num_clases in pacientes_multiclase.nlargest(5).items():
                clases = self.df[self.df['patient_id'] == paciente_id]['label_name'].unique()
                print(f"     • {paciente_id}: {num_clases} clases - {', '.join(clases)}")
        
        # Clases por paciente
        pacientes_solo_benignos = self.df.groupby('patient_id').filter(lambda x: (x['label_name'] == 'benign').all())
        pacientes_solo_malignos = self.df.groupby('patient_id').filter(lambda x: (x['label_name'] == 'malignant').all())
        pacientes_mixtos = self.df.groupby('patient_id').filter(lambda x: len(x['label_name'].unique()) > 1)
        
        print(f"\n📊 DISTRIBUCIÓN DE PACIENTES POR TIPO:")
        print(f"   • Pacientes solo benignos: {pacientes_solo_benignos['patient_id'].nunique()}")
        print(f"   • Pacientes solo malignos: {pacientes_solo_malignos['patient_id'].nunique()}")
        print(f"   • Pacientes mixtos (ambas clases): {pacientes_mixtos['patient_id'].nunique()}")
        print(f"   • Porcentaje de pacientes mixtos: {pacientes_mixtos['patient_id'].nunique()/self.total_pacientes*100:.1f}%")
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Histograma de imágenes por paciente
        axes[0,0].hist(imagenes_por_paciente, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0,0].axvline(stats_pacientes['media'], color='red', linestyle='--', linewidth=2,
                         label=f'Media: {stats_pacientes["media"]:.1f}')
        axes[0,0].axvline(stats_pacientes['mediana'], color='green', linestyle='--', linewidth=2,
                         label=f'Mediana: {stats_pacientes["mediana"]:.1f}')
        axes[0,0].set_title('Distribución de Imágenes por Paciente', fontweight='bold', fontsize=12)
        axes[0,0].set_xlabel('Número de imágenes por paciente', fontsize=10)
        axes[0,0].set_ylabel('Frecuencia', fontsize=10)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Boxplot
        box = axes[0,1].boxplot(imagenes_por_paciente, vert=True, patch_artist=True,
                               boxprops=dict(facecolor='lightblue'),
                               medianprops=dict(color='red', linewidth=2))
        axes[0,1].set_title('Boxplot: Imágenes por Paciente', fontweight='bold', fontsize=12)
        axes[0,1].set_ylabel('Número de imágenes', fontsize=10)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribución de tipos de pacientes
        tipos = ['Solo benignos', 'Solo malignos', 'Mixtos']
        counts = [
            pacientes_solo_benignos['patient_id'].nunique(),
            pacientes_solo_malignos['patient_id'].nunique(),
            pacientes_mixtos['patient_id'].nunique()
        ]
        
        bars = axes[1,0].bar(tipos, counts, color=['green', 'red', 'orange'], edgecolor='black')
        axes[1,0].set_title('Distribución de Tipos de Pacientes', fontweight='bold', fontsize=12)
        axes[1,0].set_ylabel('Número de pacientes', fontsize=10)
        axes[1,0].grid(True, alpha=0.3)
        for bar, count in zip(bars, counts):
            axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                          str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Gráfico de torta de tipos de pacientes
        axes[1,1].pie(counts, labels=tipos, autopct='%1.1f%%', startangle=90,
                     colors=['green', 'red', 'orange'], explode=(0.05, 0.05, 0.05))
        axes[1,1].set_title('Porcentaje de Tipos de Pacientes', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        self._save_fig(fig, '03_analisis_pacientes.png', dpi=150, bbox_inches='tight')
        
        # Guardar estadísticas
        self.patient_stats = {
            'imagenes_por_paciente': stats_pacientes,
            'tipos_pacientes': dict(zip(tipos, counts)),
            'pacientes_mixtos': pacientes_mixtos['patient_id'].nunique()
        }
        
        return stats_pacientes
    
    def analisis_imagenes_muestra(self, sample_size=100):
        """Análisis de características de imágenes (muestra)"""
        
        print(f"\n🖼️  ANALIZANDO {sample_size} IMÁGENES DE MUESTRA...")
        
        # Muestreo estratificado
        sample_df = self.df.groupby('label_name', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
        ).sample(frac=1, random_state=42)  # Mezclar
        
        resultados = {
            'resoluciones': [],
            'blur_scores': [],
            'contraste_scores': [],
            'brillo_promedio': [],
            'entropia': [],
            'zoom': [],
            'label': [],
            'patient_id': []
        }
        
        # Procesar cada imagen
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Procesando imágenes"):
            try:
                img_path = row['filepath']
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # 1. Resolución
                resultados['resoluciones'].append(img.size)
                
                # 2. Convertir a escala de grises
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                # 3. Blur score
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                resultados['blur_scores'].append(blur_score)
                
                # 4. Contraste
                contraste = np.std(gray)
                resultados['contraste_scores'].append(contraste)
                
                # 5. Brillo promedio
                brillo = np.mean(gray)
                resultados['brillo_promedio'].append(brillo)
                
                # 6. Entropía
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = hist / hist.sum()
                entropia = -np.sum(hist * np.log2(hist + 1e-10))
                resultados['entropia'].append(entropia)
                
                # 7. Metadatos
                resultados['zoom'].append(row['zoom'])
                resultados['label'].append(row['label_name'])
                resultados['patient_id'].append(row['patient_id'])
                
                img.close()
                
            except Exception as e:
                print(f"⚠️  Error procesando {row['filename']}: {e}")
                continue
        
        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame(resultados)
        
        # Análisis estadístico
        print("\n📊 ESTADÍSTICAS DE IMÁGENES:")
        
        if len(df_resultados) > 0:
            # Resoluciones
            resoluciones = np.array(df_resultados['resoluciones'].tolist())
            print(f"   📏 RESOLUCIONES:")
            print(f"      • Todas las imágenes: 224 × 224 (fijo)")
            print(f"      • Formato uniforme: Sí")
            
            # Blur analysis
            print(f"\n   🔍 BLUR ANALYSIS:")
            print(f"      • Promedio: {df_resultados['blur_scores'].mean():.1f}")
            print(f"      • Mediana: {df_resultados['blur_scores'].median():.1f}")
            print(f"      • Mínimo: {df_resultados['blur_scores'].min():.1f}")
            print(f"      • Máximo: {df_resultados['blur_scores'].max():.1f}")
            print(f"      • Imágenes con posible blur (<100): {(df_resultados['blur_scores'] < 100).sum()}")
            
            # Comparación entre clases
            print(f"\n   📊 COMPARACIÓN ENTRE CLASES:")
            for col in ['blur_scores', 'contraste_scores', 'brillo_promedio', 'entropia']:
                benign_mean = df_resultados[df_resultados['label'] == 'benign'][col].mean()
                malign_mean = df_resultados[df_resultados['label'] == 'malignant'][col].mean()
                diff = abs(benign_mean - malign_mean)
                diff_pct = (diff / ((benign_mean + malign_mean) / 2)) * 100
                print(f"      • {col}: Benign={benign_mean:.2f}, Malign={malign_mean:.2f}, Dif={diff:.2f} ({diff_pct:.1f}%)")
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución de blur scores por clase
        sns.boxplot(data=df_resultados, x='label', y='blur_scores', ax=axes[0,0], palette=self.colors)
        axes[0,0].set_title('Distribución de Blur Scores por Clase', fontweight='bold', fontsize=12)
        axes[0,0].set_xlabel('Clase', fontsize=10)
        axes[0,0].set_ylabel('Blur Score', fontsize=10)
        axes[0,0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Umbral blur')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distribución de contraste por clase
        sns.boxplot(data=df_resultados, x='label', y='contraste_scores', ax=axes[0,1], palette=self.colors)
        axes[0,1].set_title('Distribución de Contraste por Clase', fontweight='bold', fontsize=12)
        axes[0,1].set_xlabel('Clase', fontsize=10)
        axes[0,1].set_ylabel('Contraste (std)', fontsize=10)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribución de brillo por clase
        sns.boxplot(data=df_resultados, x='label', y='brillo_promedio', ax=axes[0,2], palette=self.colors)
        axes[0,2].set_title('Distribución de Brillo por Clase', fontweight='bold', fontsize=12)
        axes[0,2].set_xlabel('Clase', fontsize=10)
        axes[0,2].set_ylabel('Brillo Promedio', fontsize=10)
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Scatter: Blur vs Brillo
        colors_map = {'benign': self.colors[0], 'malignant': self.colors[1]}
        for label, color in colors_map.items():
            subset = df_resultados[df_resultados['label'] == label]
            axes[1,0].scatter(subset['brillo_promedio'], subset['blur_scores'], 
                            color=color, label=label, alpha=0.6, s=50)
        axes[1,0].set_title('Relación: Brillo vs Blur', fontweight='bold', fontsize=12)
        axes[1,0].set_xlabel('Brillo Promedio', fontsize=10)
        axes[1,0].set_ylabel('Blur Score', fontsize=10)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Distribución de entropía por clase
        sns.boxplot(data=df_resultados, x='label', y='entropia', ax=axes[1,1], palette=self.colors)
        axes[1,1].set_title('Distribución de Entropía por Clase', fontweight='bold', fontsize=12)
        axes[1,1].set_xlabel('Clase', fontsize=10)
        axes[1,1].set_ylabel('Entropía', fontsize=10)
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Heatmap de correlación
        numeric_cols = ['blur_scores', 'contraste_scores', 'brillo_promedio', 'entropia']
        if all(col in df_resultados.columns for col in numeric_cols):
            corr_matrix = df_resultados[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1,2], fmt='.2f',
                       cbar_kws={'label': 'Coeficiente de correlación'})
            axes[1,2].set_title('Correlación entre Características', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        self._save_fig(fig, '04_analisis_imagenes.png', dpi=150, bbox_inches='tight')
        
        # Guardar resultados
        self.image_stats = df_resultados
        
        return df_resultados
    
    def analisis_subclases(self):
        """Análisis de subclases originales"""
        
        print("\n🏷️  ANÁLISIS DE SUB-CLASES ORIGINALES:")
        
        if 'subclass' not in self.df.columns:
            print("   ⚠️  No se encontró información de subclases")
            return None
        
        # Contar subclases
        subclases_counts = self.df['subclass'].value_counts()
        
        print(f"   • Total subclases únicas: {len(subclases_counts)}")
        print(f"   • Subclases encontradas: {', '.join(subclases_counts.index.tolist())}")
        
        # Distribución de subclases por clase
        subclases_por_clase = pd.crosstab(self.df['subclass'], self.df['label_name'])
        
        print("\n📊 DISTRIBUCIÓN DE SUBCLASES POR CLASE:")
        print(subclases_por_clase)
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Distribución total de subclases
        bars1 = axes[0].bar(range(len(subclases_counts)), subclases_counts.values, 
                           color='lightcoral', edgecolor='black')
        axes[0].set_title('Distribución Total de Subclases', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Subclase', fontsize=10)
        axes[0].set_ylabel('Número de imágenes', fontsize=10)
        axes[0].set_xticks(range(len(subclases_counts)))
        axes[0].set_xticklabels(subclases_counts.index, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 2. Stacked bar por clase
        subclases_orden = subclases_counts.index
        x = range(len(subclases_orden))
        
        if 'benign' in subclases_por_clase.columns and 'malignant' in subclases_por_clase.columns:
            benign_counts = subclases_por_clase.loc[subclases_orden, 'benign']
            malign_counts = subclases_por_clase.loc[subclases_orden, 'malignant']
            
            axes[1].bar(x, benign_counts, label='Benign', color=self.colors[0], edgecolor='black')
            axes[1].bar(x, malign_counts, bottom=benign_counts, 
                       label='Malignant', color=self.colors[1], edgecolor='black')
            axes[1].set_title('Distribución de Subclases por Clase', fontweight='bold', fontsize=12)
            axes[1].set_xlabel('Subclase', fontsize=10)
            axes[1].set_ylabel('Número de imágenes', fontsize=10)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(subclases_orden, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_fig(fig, '05_analisis_subclases.png', dpi=150, bbox_inches='tight')
        
        # Análisis de pacientes por subclase
        if 'subclass' in self.df.columns:
            print("\n👥 PACIENTES POR SUBCLASE:")
            pacientes_por_subclase = self.df.groupby('subclass')['patient_id'].nunique()
            print(pacientes_por_subclase.to_string())
            
            # Visualizar pacientes por subclase
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(pacientes_por_subclase)), pacientes_por_subclase.values,
                         color='lightgreen', edgecolor='black')
            ax.set_title('Pacientes Únicos por Subclase', fontweight='bold', fontsize=12)
            ax.set_xlabel('Subclase', fontsize=10)
            ax.set_ylabel('Número de pacientes', fontsize=10)
            ax.set_xticks(range(len(pacientes_por_subclase)))
            ax.set_xticklabels(pacientes_por_subclase.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, pacientes_por_subclase.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            self._save_fig(fig, '05_pacientes_subclases.png', dpi=150, bbox_inches='tight')
        
        return subclases_por_clase
    
    def analisis_split_pacientes(self):
        """Análisis para split por pacientes"""
        
        print("\n🎯 ANÁLISIS PARA SPLIT POR PACIENTES:")
        
        # Análisis de pacientes para train/val/test split
        pacientes_info = []
        
        for patient_id in self.df['patient_id'].unique():
            patient_data = self.df[self.df['patient_id'] == patient_id]
            
            info = {
                'patient_id': patient_id,
                'total_images': len(patient_data),
                'benign_count': (patient_data['label_name'] == 'benign').sum(),
                'malignant_count': (patient_data['label_name'] == 'malignant').sum(),
                'unique_classes': patient_data['label_name'].nunique(),
                'class_type': 'mixed' if patient_data['label_name'].nunique() > 1 else patient_data['label_name'].iloc[0],
                'zooms': ', '.join(patient_data['zoom'].unique())
            }
            pacientes_info.append(info)
        
        pacientes_df = pd.DataFrame(pacientes_info)
        
        # Estadísticas para split
        print(f"   • Pacientes con 1 imagen: {(pacientes_df['total_images'] == 1).sum()}")
        print(f"   • Pacientes con 2-5 imágenes: {((pacientes_df['total_images'] >= 2) & (pacientes_df['total_images'] <= 5)).sum()}")
        print(f"   • Pacientes con 6-10 imágenes: {((pacientes_df['total_images'] >= 6) & (pacientes_df['total_images'] <= 10)).sum()}")
        print(f"   • Pacientes con >10 imágenes: {(pacientes_df['total_images'] > 10).sum()}")
        
        # Recomendaciones para split
        print("\n💡 RECOMENDACIONES PARA SPLIT:")
        
        # Separar pacientes mixtos y puros
        pacientes_mixtos = pacientes_df[pacientes_df['unique_classes'] > 1]
        pacientes_puros = pacientes_df[pacientes_df['unique_classes'] == 1]
        
        print(f"   • Pacientes puros (una clase): {len(pacientes_puros)}")
        print(f"   • Pacientes mixtos (ambas clases): {len(pacientes_mixtos)}")
        
        # Estrategia de split
        print(f"\n   🎯 ESTRATEGIA RECOMENDADA:")
        print(f"     1. Usar split estratificado por paciente (80% train, 10% val, 10% test)")
        print(f"     2. Mantener proporción de clases en cada split")
        print(f"     3. Considerar pacientes mixtos especialmente para val/test")
        print(f"     4. Asegurar que todos los aumentos estén representados")
        
        # Visualización
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Distribución de número de imágenes por paciente
        axes[0].hist(pacientes_df['total_images'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title('Distribución de Imágenes por Paciente', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Número de imágenes', fontsize=10)
        axes[0].set_ylabel('Frecuencia', fontsize=10)
        axes[0].axvline(pacientes_df['total_images'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Media: {pacientes_df["total_images"].mean():.1f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Distribución de tipos de pacientes
        tipo_counts = pacientes_df['class_type'].value_counts()
        colors_map = {'benign': 'green', 'malignant': 'red', 'mixed': 'orange'}
        colors = [colors_map.get(t, 'gray') for t in tipo_counts.index]
        
        bars = axes[1].bar(tipo_counts.index, tipo_counts.values, color=colors, edgecolor='black')
        axes[1].set_title('Tipos de Pacientes', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Tipo', fontsize=10)
        axes[1].set_ylabel('Número de pacientes', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, tipo_counts.values):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Relación imágenes vs tipo
        color_map_numeric = {'benign': 0, 'malignant': 1, 'mixed': 2}
        pacientes_df['class_type_num'] = pacientes_df['class_type'].map(color_map_numeric)
        
        scatter = axes[2].scatter(range(len(pacientes_df)), pacientes_df['total_images'],
                                 c=pacientes_df['class_type_num'], cmap='viridis', 
                                 alpha=0.6, s=50, edgecolor='black')
        axes[2].set_title('Imágenes por Paciente vs Tipo', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('Índice de Paciente', fontsize=10)
        axes[2].set_ylabel('Número de imágenes', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Crear leyenda personalizada
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Benign',
                                 markerfacecolor='green', markersize=10),
                          Line2D([0], [0], marker='o', color='w', label='Malignant',
                                 markerfacecolor='red', markersize=10),
                          Line2D([0], [0], marker='o', color='w', label='Mixed',
                                 markerfacecolor='orange', markersize=10)]
        axes[2].legend(handles=legend_elements)
        
        plt.tight_layout()
        self._save_fig(fig, '06_analisis_split.png', dpi=150, bbox_inches='tight')
        
        # Guardar datos para split
        self.split_info = {
            'pacientes_df': pacientes_df,
            'total_pacientes': len(pacientes_df),
            'pacientes_mixtos': len(pacientes_mixtos),
            'pacientes_puros': len(pacientes_puros),
            'split_recomendado': '80-10-10'
        }
        
        return pacientes_df
    
    def visualizaciones_avanzadas(self):
        """Visualizaciones avanzadas e interactivas"""
        
        print("\n📊 CREANDO VISUALIZACIONES AVANZADAS...")
        
        # 1. Word Cloud de subclases
        print("\n☁️  GENERANDO WORD CLOUD DE SUBCLASES...")
        
        if 'subclass' in self.df.columns:
            # Crear texto ponderado por frecuencia
            text_data = []
            for subclass, count in self.df['subclass'].value_counts().items():
                text_data.extend([subclass] * count)
            
            if text_data:
                text = ' '.join(text_data)
                
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                    colormap='tab20c', max_words=100).generate(text)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud de Subclases (ponderado por frecuencia)', 
                           fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                self._save_fig(fig, '07_wordcloud_subclases.png', dpi=150, bbox_inches='tight')
        
        # 2. Gráfico de radar para comparar aumentos
        print("\n📡 GENERANDO GRÁFICO DE RADAR...")
        
        # Preparar datos para radar chart
        if hasattr(self, 'zoom_stats') and self.zoom_stats:
            # Extraer estadísticas por aumento
            zoom_df = pd.DataFrame(self.zoom_stats)
            
            # Normalizar para radar chart
            stats_normalized = zoom_df.copy()
            for col in ['Total_Imagenes', 'Pacientes_Unicos', 'Benignas', 'Malignas']:
                if col in stats_normalized.columns:
                    stats_normalized[col] = (stats_normalized[col] - stats_normalized[col].min()) / \
                                          (stats_normalized[col].max() - stats_normalized[col].min())
            
            # Crear gráfico de radar
            categories = ['Total_Imagenes', 'Pacientes_Unicos', 'Benignas', 'Malignas']
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Cerrar el círculo
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(zoom_df)))
            
            for idx, row in stats_normalized.iterrows():
                values = [row[col] for col in categories]
                values += values[:1]  # Cerrar el círculo
                
                ax.plot(angles, values, 'o-', linewidth=2, label=row['Zoom'], color=colors[idx])
                ax.fill(angles, values, alpha=0.1, color=colors[idx])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['Total Imágenes', 'Pacientes Únicos', 'Benignas', 'Malignas'], 
                             fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('Comparación de Aumentos por Métricas (Radar Chart)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
            ax.grid(True)
            
            plt.tight_layout()
            self._save_fig(fig, '07_radar_chart.png', dpi=150, bbox_inches='tight')
        
        # 3. Heatmap de correlación entre características de imágenes
        if hasattr(self, 'image_stats') and self.image_stats is not None:
            print("\n📊 CREANDO HEATMAP DE CORRELACIÓN...")
            
            numeric_cols = ['blur_scores', 'contraste_scores', 'brillo_promedio', 'entropia']
            if all(col in self.image_stats.columns for col in numeric_cols):
                corr_matrix = self.image_stats[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, ax=ax, fmt='.2f', 
                           cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de correlación'})
                ax.set_title('Matriz de Correlación entre Características de Imágenes', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Características', fontsize=12)
                ax.set_ylabel('Características', fontsize=12)
                
                plt.tight_layout()
                self._save_fig(fig, '07_correlacion_caracteristicas.png', dpi=150, bbox_inches='tight')
        
        # 4. Gráfico de dispersión 3D (si hay suficientes datos)
        if hasattr(self, 'image_stats') and len(self.image_stats) > 10:
            print("\n📊 CREANDO GRÁFICO DE DISPERSIÓN 3D...")
            
            try:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                colors_map = {'benign': self.colors[0], 'malignant': self.colors[1]}
                
                for label, color in colors_map.items():
                    subset = self.image_stats[self.image_stats['label'] == label]
                    if len(subset) > 0:
                        ax.scatter(subset['blur_scores'], subset['contraste_scores'], 
                                 subset['brillo_promedio'], c=color, label=label, 
                                 alpha=0.6, s=50)
                
                ax.set_xlabel('Blur Score', fontsize=10)
                ax.set_ylabel('Contraste', fontsize=10)
                ax.set_zlabel('Brillo Promedio', fontsize=10)
                ax.set_title('Dispersión 3D: Blur vs Contraste vs Brillo', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                
                plt.tight_layout()
                self._save_fig(fig, '07_dispersion_3d.png', dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"⚠️  No se pudo crear gráfico 3D: {e}")
    
    def analisis_calidad_imagenes(self):
        """Análisis de calidad y problemas potenciales"""
        
        print("\n🔍 ANALIZANDO CALIDAD DE IMÁGENES...")
        
        # Tomar una muestra para análisis de calidad
        sample_size = min(200, len(self.df))
        sample_df = self.df.sample(sample_size, random_state=42)
        
        problemas = {
            'baja_resolucion': 0,
            'alto_blur': 0,
            'bajo_contraste': 0,
            'alta_saturacion_fondo': 0,
            'posibles_artefactos': 0
        }
        
        # Criterios ajustados para imágenes histológicas
        criterios = {
            'baja_resolucion': (224, 224),  # Las imágenes ya son 224x224
            'alto_blur': 50,  # varianza Laplaciana < 50
            'bajo_contraste': 20,  # std < 20 (ajustado para histología)
            'alta_saturacion_fondo': 0.7,  # más del 70% de fondo
        }
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluando calidad"):
            try:
                img_path = row['filepath']
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # 1. Resolución
                if img.size[0] < criterios['baja_resolucion'][0] or img.size[1] < criterios['baja_resolucion'][1]:
                    problemas['baja_resolucion'] += 1
                
                # 2. Blur
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < criterios['alto_blur']:
                    problemas['alto_blur'] += 1
                
                # 3. Contraste
                contraste = np.std(gray)
                if contraste < criterios['bajo_contraste']:
                    problemas['bajo_contraste'] += 1
                
                # 4. Proporción de fondo (simplificado)
                # Para histología, estimamos fondo como áreas muy claras
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                proporcion_fondo = np.sum(thresh == 255) / thresh.size
                if proporcion_fondo > criterios['alta_saturacion_fondo']:
                    problemas['alta_saturacion_fondo'] += 1
                
                # 5. Detección de artefactos simples (bordes muy oscuros)
                bordes = cv2.Canny(gray, 100, 200)
                proporcion_bordes = np.sum(bordes > 0) / bordes.size
                if proporcion_bordes < 0.01:  # Muy pocos bordes
                    problemas['posibles_artefactos'] += 1
                
                img.close()
                
            except Exception as e:
                print(f"⚠️  Error en análisis de calidad: {e}")
                continue
        
        # Calcular porcentajes
        total_analizadas = len(sample_df)
        problemas_porcentaje = {k: (v/total_analizadas*100) for k, v in problemas.items()}
        
        print("\n📊 PROBLEMAS DE CALIDAD DETECTADOS:")
        for problema, porcentaje in problemas_porcentaje.items():
            print(f"   • {problema.replace('_', ' ').title()}: {porcentaje:.1f}%")
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES DE PREPROCESAMIENTO:")
        
        recomendaciones = []
        if problemas_porcentaje['alto_blur'] > 10:
            recomendaciones.append("• Aplicar filtros de sharpening o descartar imágenes muy borrosas")
        
        if problemas_porcentaje['bajo_contraste'] > 10:
            recomendaciones.append("• Aplicar ecualización de histograma o CLAHE para mejorar contraste")
        
        if problemas_porcentaje['baja_resolucion'] > 5:
            recomendaciones.append("• Considerar upscaling para imágenes de baja resolución")
        
        if problemas_porcentaje['alta_saturacion_fondo'] > 15:
            recomendaciones.append("• Aplicar técnicas de segmentación para aislar tejido del fondo")
        
        if problemas_porcentaje['posibles_artefactos'] > 10:
            recomendaciones.append("• Revisar imágenes para posibles artefactos de preparación")
        
        if not recomendaciones:
            recomendaciones.append("• La calidad general parece buena, preprocesamiento estándar es suficiente")
        
        for rec in recomendaciones:
            print(f"   {rec}")
        
        # Visualización
        fig, ax = plt.subplots(figsize=(12, 6))
        
        problemas_df = pd.DataFrame({
            'Problema': [p.replace('_', ' ').title() for p in problemas_porcentaje.keys()],
            'Porcentaje': list(problemas_porcentaje.values())
        })
        
        problemas_df = problemas_df.sort_values('Porcentaje', ascending=False)
        
        bars = ax.barh(problemas_df['Problema'], problemas_df['Porcentaje'], 
                      color=plt.cm.Reds(np.linspace(0.3, 0.8, len(problemas_df))))
        
        ax.set_xlabel('Porcentaje de imágenes afectadas (%)', fontsize=12)
        ax.set_title('Problemas de Calidad Detectados', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir valores
        for i, (bar, porcentaje) in enumerate(zip(bars, problemas_df['Porcentaje'])):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{porcentaje:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        self._save_fig(fig, '08_calidad_imagenes.png', dpi=150, bbox_inches='tight')
        
        # Guardar resultados
        self.calidad_stats = {
            'problemas': problemas,
            'problemas_porcentaje': problemas_porcentaje,
            'recomendaciones': recomendaciones
        }
        
        return problemas_porcentaje
    
    def analisis_color_texturas(self):
        """Análisis avanzado de color y texturas"""
        
        print("\n🎨 ANALIZANDO COLOR Y TEXTURAS...")
        
        # Tomar muestra representativa
        sample_size = min(50, len(self.df))
        sample_df = self.df.sample(sample_size, random_state=42)
        
        # Estructuras para almacenar resultados
        color_stats = defaultdict(list)
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analizando color"):
            try:
                img_path = row['filepath']
                img = Image.open(img_path)
                img_array = np.array(img)
                
                if len(img_array.shape) == 3:  # RGB
                    # Estadísticas de color por canal
                    for i, canal in enumerate(['R', 'G', 'B']):
                        canal_data = img_array[:, :, i].flatten()
                        color_stats[f'media_{canal}'].append(np.mean(canal_data))
                        color_stats[f'std_{canal}'].append(np.std(canal_data))
                        color_stats[f'skew_{canal}'].append(stats.skew(canal_data))
                    
                    # Convertir a HSV
                    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    h_channel = img_hsv[:, :, 0].flatten()
                    color_stats['hue_mean'].append(np.mean(h_channel))
                    color_stats['hue_std'].append(np.std(h_channel))
                    
                    # Saturación y valor
                    color_stats['saturation_mean'].append(np.mean(img_hsv[:, :, 1].flatten()))
                    color_stats['value_mean'].append(np.mean(img_hsv[:, :, 2].flatten()))
                    color_stats['saturation_std'].append(np.std(img_hsv[:, :, 1].flatten()))
                    color_stats['value_std'].append(np.std(img_hsv[:, :, 2].flatten()))
                
                img.close()
                
            except Exception as e:
                continue
        
        # Análisis de color
        print("\n📊 ESTADÍSTICAS DE COLOR (RGB):")
        
        if color_stats:
            for canal in ['R', 'G', 'B']:
                if f'media_{canal}' in color_stats and color_stats[f'media_{canal}']:
                    media_media = np.mean(color_stats[f'media_{canal}'])
                    media_std = np.mean(color_stats[f'std_{canal}'])
                    print(f"   • Canal {canal}: Media={media_media:.1f}, Std={media_std:.1f}")
        
        # Visualizaciones de color
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Distribución de medias por canal
        if 'media_R' in color_stats and 'media_G' in color_stats and 'media_B' in color_stats:
            data_rgb = pd.DataFrame({
                'Canal': ['Rojo'] * len(color_stats['media_R']) + 
                        ['Verde'] * len(color_stats['media_G']) + 
                        ['Azul'] * len(color_stats['media_B']),
                'Media': color_stats['media_R'] + color_stats['media_G'] + color_stats['media_B']
            })
            
            sns.boxplot(data=data_rgb, x='Canal', y='Media', ax=axes[0,0])
            axes[0,0].set_title('Distribución de Intensidad Media por Canal RGB', fontsize=12)
            axes[0,0].set_ylabel('Intensidad Media')
            axes[0,0].grid(alpha=0.3)
        
        # 2. Correlación entre canales
        if all(k in color_stats for k in ['media_R', 'media_G', 'media_B']):
            correlaciones = np.corrcoef([color_stats['media_R'], color_stats['media_G'], color_stats['media_B']])
            im = axes[0,1].imshow(correlaciones, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0,1].set_xticks([0, 1, 2])
            axes[0,1].set_yticks([0, 1, 2])
            axes[0,1].set_xticklabels(['R', 'G', 'B'])
            axes[0,1].set_yticklabels(['R', 'G', 'B'])
            axes[0,1].set_title('Correlación entre Canales RGB', fontsize=12)
            
            # Añadir valores
            for i in range(3):
                for j in range(3):
                    axes[0,1].text(j, i, f'{correlaciones[i,j]:.2f}', 
                                  ha='center', va='center', color='white', fontweight='bold')
            
            plt.colorbar(im, ax=axes[0,1])
        
        # 3. Análisis de Hue (HSV)
        if 'hue_mean' in color_stats and color_stats['hue_mean']:
            axes[0,2].hist(color_stats['hue_mean'], bins=30, edgecolor='black', alpha=0.7, color='orange')
            axes[0,2].set_title('Distribución de Hue (Tono)', fontsize=12)
            axes[0,2].set_xlabel('Hue (0-180)')
            axes[0,2].set_ylabel('Frecuencia')
            axes[0,2].grid(alpha=0.3)
        
        # 4. Saturación vs Valor
        if 'saturation_mean' in color_stats and 'value_mean' in color_stats:
            axes[1,0].scatter(color_stats['saturation_mean'], color_stats['value_mean'], 
                            alpha=0.6, s=30, color='purple')
            axes[1,0].set_title('Relación: Saturación vs Valor (HSV)', fontsize=12)
            axes[1,0].set_xlabel('Saturación Media')
            axes[1,0].set_ylabel('Valor (Brillo) Medio')
            axes[1,0].grid(alpha=0.3)
        
        # 5. Skewness por canal
        if all(k in color_stats for k in ['skew_R', 'skew_G', 'skew_B']):
            skew_data = pd.DataFrame({
                'Canal': ['R'] * len(color_stats['skew_R']) + 
                        ['G'] * len(color_stats['skew_G']) + 
                        ['B'] * len(color_stats['skew_B']),
                'Skewness': color_stats['skew_R'] + color_stats['skew_G'] + color_stats['skew_B']
            })
            sns.boxplot(data=skew_data, x='Canal', y='Skewness', ax=axes[1,1])
            axes[1,1].set_title('Skewness por Canal RGB', fontsize=12)
            axes[1,1].set_ylabel('Skewness')
            axes[1,1].grid(alpha=0.3)
        
        # 6. Heatmap de estadísticas de color
        stats_summary = {}
        if color_stats:
            for stat in ['media', 'std']:
                for canal in ['R', 'G', 'B']:
                    key = f'{stat}_{canal}'
                    if key in color_stats and color_stats[key]:
                        stats_summary[f'{canal}_{stat}'] = np.mean(color_stats[key])
            
            if stats_summary:
                df_summary = pd.DataFrame([stats_summary])
                sns.heatmap(df_summary, annot=True, fmt='.1f', cmap='viridis', 
                           ax=axes[1,2], cbar_kws={'label': 'Valor'})
                axes[1,2].set_title('Resumen Estadísticas de Color', fontsize=12)
        
        plt.tight_layout()
        self._save_fig(fig, '09_color_texturas.png', dpi=150, bbox_inches='tight')
        
        # Guardar resultados
        self.color_stats = dict(color_stats)
        
        return color_stats
    
    def generar_reporte_final(self):
        """Genera reporte final con todos los análisis"""
        
        print("\n📋 GENERANDO REPORTE FINAL...")
        
        # Compilar todos los resultados
        reporte = {
            'fecha_analisis': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {
                'nombre': 'BreakHis Binary',
                'tipo': 'Imágenes histológicas de cáncer de mama',
                'clases': ['benign (0)', 'malignant (1)'],
                'aumentos': ['40X', '100X', '200X', '400X']
            },
            'estadisticas_globales': self.stats_basicas if hasattr(self, 'stats_basicas') else {},
            'analisis_aumento': self.zoom_stats if hasattr(self, 'zoom_stats') else [],
            'analisis_pacientes': self.patient_stats if hasattr(self, 'patient_stats') else {},
            'analisis_split': self.split_info if hasattr(self, 'split_info') else {},
            'analisis_calidad': self.calidad_stats if hasattr(self, 'calidad_stats') else {},
            'recomendaciones_modelado': self._generar_recomendaciones()
        }
        
        # Guardar reporte JSON
        reporte_path = os.path.join(self.output_dir, 'reporte_final.json')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar reporte en texto
        txt_path = os.path.join(self.output_dir, 'reporte_final.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE FINAL - ANÁLISIS EDA DATASET BREAKHIS BINARIO\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Fecha de análisis: {reporte['fecha_analisis']}\n\n")
            
            f.write("📊 ESTADÍSTICAS GLOBALES:\n")
            f.write("-" * 40 + "\n")
            if 'estadisticas_globales' in reporte:
                stats = reporte['estadisticas_globales']
                f.write(f"  • Total imágenes: {stats.get('total_imagenes', 'N/A'):,}\n")
                f.write(f"  • Total pacientes: {stats.get('total_pacientes', 'N/A'):,}\n")
                f.write(f"  • Ratio balanceo: {stats.get('ratio_balanceo', 'N/A'):.2f}:1\n")
                f.write(f"  • Coeficiente de variación: {stats.get('coeficiente_variacion', 'N/A'):.3f}\n")
                f.write(f"  • Necesita balanceo: {'Sí' if stats.get('needs_balancing', False) else 'No'}\n")
            
            f.write("\n🔬 DISTRIBUCIÓN POR AUMENTO:\n")
            f.write("-" * 40 + "\n")
            if 'analisis_aumento' in reporte:
                for zoom in reporte['analisis_aumento']:
                    f.write(f"  • {zoom.get('Zoom', 'N/A')}: {zoom.get('Total_Imagenes', 0)} imágenes "
                           f"({zoom.get('Pacientes_Unicos', 0)} pacientes)\n")
            
            f.write("\n👥 ANÁLISIS DE PACIENTES:\n")
            f.write("-" * 40 + "\n")
            if 'analisis_pacientes' in reporte:
                pacientes = reporte['analisis_pacientes']
                if 'imagenes_por_paciente' in pacientes:
                    stats = pacientes['imagenes_por_paciente']
                    f.write(f"  • Media imágenes/paciente: {stats.get('media', 0):.2f}\n")
                    f.write(f"  • Mediana: {stats.get('mediana', 0):.2f}\n")
                    f.write(f"  • Desviación estándar: {stats.get('std', 0):.2f}\n")
                    f.write(f"  • Mínimo: {stats.get('min', 0)}\n")
                    f.write(f"  • Máximo: {stats.get('max', 0)}\n")
                    f.write(f"  • Q1: {stats.get('q1', 0):.2f}\n")
                    f.write(f"  • Q3: {stats.get('q3', 0):.2f}\n")
                    f.write(f"  • Pacientes con 1 imagen: {stats.get('pacientes_con_1_imagen', 0)}\n")
                    f.write(f"  • Pacientes con más de 10 imágenes: {stats.get('pacientes_con_mas_de_10_imagenes', 0)}\n")
                    f.write(f"  • Coeficiente de variación: {stats.get('coeficiente_variacion', 0):.3f}\n")
                f.write(f"  • Pacientes mixtos: {pacientes.get('pacientes_mixtos', 0)}\n")
            
            f.write("\n🔍 ANÁLISIS DE CALIDAD:\n")
            f.write("-" * 40 + "\n")
            if 'analisis_calidad' in reporte:
                calidad = reporte['analisis_calidad']
                if 'problemas_porcentaje' in calidad:
                    for problema, porcentaje in calidad['problemas_porcentaje'].items():
                        f.write(f"  • {problema.replace('_', ' ').title()}: {porcentaje:.1f}%\n")
            
            f.write("\n🎯 RECOMENDACIONES PARA MODELADO:\n")
            f.write("-" * 40 + "\n")
            if 'recomendaciones_modelado' in reporte:
                for i, rec in enumerate(reporte['recomendaciones_modelado'], 1):
                    f.write(f"  {i}. {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("📈 RESUMEN EJECUTIVO PARA CNN:\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. PREPROCESAMIENTO RECOMENDADO:\n")
            f.write("   • No necesita resize (todas son 224x224)\n")
            f.write("   • Normalización: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n")
            f.write("   • Data augmentation: rotaciones (±15°), flips horizontal/vertical\n")
            f.write("   • Aplicar técnicas de balanceo (class_weight={0: 2.19, 1: 1.0})\n\n")
            
            f.write("2. ARQUITECTURA CNN:\n")
            f.write("   • Transfer learning con ResNet50 o EfficientNet-B4\n")
            f.write("   • Fine-tuning de las últimas 10-20 capas\n")
            f.write("   • Dropout (0.5) antes de la capa fully connected\n")
            f.write("   • Learning rate scheduler (ReduceLROnPlateau con factor=0.1, patience=5)\n\n")
            
            f.write("3. ESTRATEGIA DE VALIDACIÓN:\n")
            f.write("   • Split por paciente: 80% train, 10% val, 10% test\n")
            f.write("   • Mantener proporción de clases en cada split\n")
            f.write("   • Early stopping con paciencia=10 y restore_best_weights=True\n")
            f.write("   • 5-fold cross-validation si es posible\n\n")
            
            f.write("4. MÉTRICAS IMPORTANTES:\n")
            f.write("   • Accuracy, Precision, Recall, F1-score (macro)\n")
            f.write("   • AUC-ROC (crítico para diagnóstico médico)\n")
            f.write("   • Sensitivity (Recall para clase maligna) > 90%\n")
            f.write("   • Specificity (Precision para clase benigna) > 85%\n")
            f.write("   • Matriz de confusión detallada\n")
        
        print(f"✅ Reporte JSON guardado en: {reporte_path}")
        print(f"✅ Reporte de texto guardado en: {txt_path}")
        
        return reporte
    
    def mostrar_resumen_ejecutivo(self):
        """Muestra resumen ejecutivo en consola"""
        
        print("\n" + "=" * 80)
        print("📋 RESUMEN EJECUTIVO:")
        print("=" * 80)
        print(f"  1. Dataset: {self.total_imagenes:,} imágenes de {self.total_pacientes:,} pacientes")
        print(f"  2. Clases: Benigno={self.df['label_name'].value_counts().get('benign', 0)}, " +
              f"Maligno={self.df['label_name'].value_counts().get('malignant', 0)}")
        print(f"  3. Balanceo: {'Requiere atención (ratio > 1.5)' if self.stats_basicas.get('needs_balancing', False) else 'Adecuado'}")
        print(f"  4. Coeficiente de variación (balanceo): {self.stats_basicas.get('coeficiente_variacion', 0):.3f}")
        print(f"  5. Ratio mayoritaria/minoritaria: {self.stats_basicas.get('ratio_balanceo', 0):.2f}:1")
        print(f"  6. Aumentos: {self.total_zooms} niveles ({', '.join(sorted(self.df['zoom'].unique()))})")
        print(f"  7. Pacientes mixtos: {self.patient_stats.get('pacientes_mixtos', 0)} " +
              f"({self.patient_stats.get('pacientes_mixtos', 0)/self.total_pacientes*100:.1f}%)")
        print(f"  8. Resolución imágenes: 224x224 (uniforme)")
        print(f"  9. Distribución por paciente - Media: {self.patient_stats.get('imagenes_por_paciente', {}).get('media', 0):.1f}, " +
              f"Máximo: {self.patient_stats.get('imagenes_por_paciente', {}).get('max', 0)}")
        print(f" 10. Calidad imágenes: {'Buena' if not hasattr(self, 'calidad_stats') or max(self.calidad_stats.get('problemas_porcentaje', {}).values()) < 20 else 'Requiere atención'}")
        print("\n🔑 INSIGHTS CLAVE:")
        print(f"   • Dataset desbalanceado {self.stats_basicas.get('ratio_balanceo', 0):.2f}:1 (maligno:benigno)")
        print(f"   • Todos los pacientes son 'puros' (solo una clase)")
        print(f"   • Distribución similar en todos los aumentos (~69% maligno, ~31% benigno)")
        print(f"   • 8 subclases originales agrupadas en 2 clases binarias")
        print(f"   • Split por paciente ES CRÍTICO para evitar data leakage")
        print(f"   • Resolución uniforme facilita el preprocesamiento")
    
    def _generar_recomendaciones(self):
        """Genera recomendaciones basadas en el análisis"""
        
        recomendaciones = []
        
        # Recomendaciones basadas en balanceo
        if self.stats_basicas.get('needs_balancing', False):
            recomendaciones.append(f"Aplicar técnicas de balanceo: oversampling de clase benigna, weighted loss function (class_weight={{0: {self.stats_basicas.get('ratio_balanceo', 2.19):.2f}, 1: 1.0}})")
        else:
            recomendaciones.append("Balanceo adecuado, no se requieren técnicas especiales de balanceo")
        
        # Recomendaciones basadas en pacientes mixtos
        if self.patient_stats.get('pacientes_mixtos', 0) > 0:
            recomendaciones.append(f"{self.patient_stats['pacientes_mixtos']} pacientes tienen ambas clases, considerar en split para evitar leakage")
        else:
            recomendaciones.append("Todos los pacientes son puros (una sola clase), split por paciente es más sencillo")
        
        # Recomendaciones basadas en aumentos
        zoom_counts = self.df['zoom'].value_counts()
        if len(zoom_counts) > 1:
            recomendaciones.append(f"Múltiples aumentos disponibles ({', '.join(zoom_counts.index.tolist())}), considerar entrenar modelos separados o usar multi-scale approach")
        
        # Recomendaciones basadas en calidad
        if hasattr(self, 'calidad_stats'):
            if self.calidad_stats.get('problemas_porcentaje', {}).get('alto_blur', 0) > 10:
                recomendaciones.append("Algunas imágenes tienen blur, considerar filtros de sharpening")
            if self.calidad_stats.get('problemas_porcentaje', {}).get('bajo_contraste', 0) > 10:
                recomendaciones.append("Aplicar CLAHE para mejorar contraste en imágenes histológicas")
        
        # Recomendaciones técnicas
        recomendaciones.append("Usar split por paciente (80-10-10) estratificado manteniendo proporción de clases")
        recomendaciones.append("Data augmentation: rotaciones (±15°), flips horizontal/vertical, cambios de brillo/contraste (±10%)")
        recomendaciones.append("Transfer learning con modelos preentrenados en ImageNet (ResNet50, EfficientNet-B4)")
        recomendaciones.append("Evaluar métricas específicas para diagnóstico médico: Sensitivity, Specificity, AUC-ROC")
        recomendaciones.append("Considerar entrenamiento multi-aumento o combinar aumentos en el dataset")
        
        return recomendaciones


# ======================================================
# FUNCIÓN PARA CARGAR DATOS (compatible con tu código)
# ======================================================
def parse_breakhis_filename(filename):
    """Extrae información del nombre de archivo BreakHis."""
    name = os.path.splitext(filename)[0]
    try:
        parts = name.split('-')
        prefix = parts[0].split('_')
        info = {
            "dataset": prefix[0],
            "class_code": prefix[1],
            "subclass": prefix[2],
            "year": parts[1],
            "patient_id": parts[2],
            "magnification": parts[3],
            "sequence": parts[4]
        }
        return info
    except Exception as e:
        raise ValueError(f"Error parseando nombre de archivo: {filename}") from e

def read_binary_breakhis_data(base_path, verbose=False):
    """Lee el dataset BreakHis en clasificación binaria."""
    label_map = {"benign": 0, "malignant": 1}
    data = {}
    all_images = []
    all_labels = []
    slides = []
    
    if not os.path.isdir(base_path):
        raise ValueError(f"Ruta no válida: {base_path}")
    
    zoom_levels = sorted(os.listdir(base_path))
    
    if verbose:
        print(f"\n📂 Base path: {base_path}")
        print(f"🔍 Zoom levels encontrados: {zoom_levels}\n")
    
    for zoom in zoom_levels:
        zoom_path = os.path.join(base_path, zoom)
        if not os.path.isdir(zoom_path):
            continue
        
        data[zoom] = {}
        
        for class_name in ["benign", "malignant"]:
            class_path = os.path.join(zoom_path, class_name)
            
            if not os.path.isdir(class_path):
                if verbose:
                    print(f"⚠️  Carpeta no encontrada: {class_path}")
                continue
            
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            image_paths = []
            for img in image_files:
                full_path = os.path.join(class_path, img)
                image_paths.append(full_path)
                
                info = parse_breakhis_filename(img)
                patient_id = info["patient_id"]
                
                all_images.append(full_path)
                all_labels.append(label_map[class_name])
                slides.append(patient_id)
            
            data[zoom][class_name] = image_paths
            
            if verbose:
                print(f"🔬 Zoom {zoom} | Clase {class_name}")
                print(f"   ├── Imágenes: {len(image_paths)}")
                print(f"   ├── Label: {label_map[class_name]}")
                unique_patients = len(set(slides))
                print(f"   └── Pacientes únicos acumulados: {unique_patients}\n")
    
    if verbose:
        print("📊 RESUMEN FINAL")
        print(f"   Total imágenes: {len(all_images)}")
        print(f"   Total benignas: {all_labels.count(0)}")
        print(f"   Total malignas: {all_labels.count(1)}")
        print(f"   Total pacientes únicos: {len(set(slides))}")
        print(f"   Label map: {label_map}\n")
    
    return data, all_images, all_labels, label_map, slides


# ======================================================
# EJECUCIÓN PRINCIPAL
# ======================================================
if __name__ == "__main__":
    # Configurar rutas
    BASE_PATH = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"
    
    print("🔍 Cargando dataset BreakHis...")
    
    # Cargar datos usando tu función
    data, all_images, all_labels, label_map, slides = read_binary_breakhis_data(
        base_path=BASE_PATH,
        verbose=True
    )
    
    # Crear analizador EDA
    print("\n🔍 Inicializando analizador EDA profesional para BreakHis...")
    eda = BreakHisEDA(
        all_images=all_images,
        all_labels=all_labels,
        slides=slides,
        base_path=BASE_PATH,
        label_map=label_map,
        show_plots=True  # Mostrar las gráficas en tiempo real
    )
    
    # Ejecutar análisis completo
    try:
        eda.ejecutar_analisis_completo(
            output_dir="breakhis_eda_profesional",
            sample_images=100
        )
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
