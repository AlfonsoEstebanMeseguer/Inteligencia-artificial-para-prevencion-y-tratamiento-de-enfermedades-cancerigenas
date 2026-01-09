import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
try:
    from skimage.feature import greycomatrix, greycoprops
    SKIMAGE_AVAILABLE=True
except Exception:
    SKIMAGE_AVAILABLE=False

# Paleta y orden de aumentos fijos para todas las graficas
PALETTE=['#4C72B0','#DD8452']
ZOOM_ORDER=['40X','100X','200X','400X']

# Agrega medias y desviaciones por canal segun agrupacion
def _agg_color(df_color,group_col):
    res={}
    for key,sub in df_color.groupby(group_col):
        res[key]={ch:{'mean':float(sub[ch].mean()),'std':float(sub[ch].std())} for ch in ['R','G','B']}
    return res

# Path base configurable via variable de entorno para portabilidad
base_path=os.environ.get('BREAKHIS_BASE',os.path.join(os.getcwd(),'BreakHist_Binary','BreakHist','data','BreakHis - Breast Cancer Histopathological Database','dataset_cancer_v1','dataset_cancer_v1','classificacao_binaria'))
# Numero de imagenes a muestrear en el analisis de imagenes
sample_images=100
# Mapeo de labels de la base BreakHis
label_map={'benign':0,'malignant':1}
data={}
all_images=[]
all_labels=[]
slides=[]

if not os.path.isdir(base_path):
    raise ValueError(f'Ruta no valida: {base_path}')

# Recorrido del arbol de zooms y clases, cargando paths y labels
zoom_levels=sorted(os.listdir(base_path))
print(f'Zoom levels encontrados: {zoom_levels}')
for zoom in zoom_levels:
    zoom_path=os.path.join(base_path,zoom)
    if not os.path.isdir(zoom_path):
        continue
    data[zoom]={}
    for class_name in ['benign','malignant']:
        class_path=os.path.join(zoom_path,class_name)
        if not os.path.isdir(class_path):
            print(f'Carpeta no encontrada: {class_path}')
            continue
        image_files=[f for f in os.listdir(class_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        image_paths=[]
        for img in image_files:
            full_path=os.path.join(class_path,img)
            image_paths.append(full_path)
            name=os.path.splitext(img)[0]
            parts=name.split('-')
            prefix=parts[0].split('_')
            if len(parts)>2:
                patient_id=parts[2]
            else:
                patient_id='NA'
            all_images.append(full_path)
            all_labels.append(label_map[class_name])
            slides.append(patient_id)
        data[zoom][class_name]=image_paths
        print(f'Zoom {zoom} | Clase {class_name}')
        print(f'Imagenes: {len(image_paths)}')
        print(f'Label: {label_map[class_name]}')
        print(f'Pacientes unicos acumulados: {len(set(slides))}')
print('RESUMEN FINAL')
print(f'Total imagenes: {len(all_images)}')
print(f'Total benignas: {all_labels.count(0)}')
print(f'Total malignas: {all_labels.count(1)}')
print(f'Total pacientes unicos: {len(set(slides))}')
print(f'Label map: {label_map}')

# Preparar dataframe con metadata derivada del nombre de archivo
registros=[]
for img_path,label,slide in tqdm(zip(all_images,all_labels,slides),total=len(all_images),desc='Procesando',disable=True):
    try:
        parts=img_path.split(os.sep)
        zoom=parts[-3]
        if label==0:
            class_name='benign'
        else:
            class_name='malignant'
        filename=os.path.basename(img_path)
        name_parts=filename.split('-')
        base_tokens=name_parts[0].split('_')
        dataset=base_tokens[0]
        class_code=base_tokens[1]
        subclass=base_tokens[2]
        year=name_parts[1]
        sequence=name_parts[4]
        registros.append({'filepath':img_path,'filename':filename,'label':label,'label_name':class_name,'patient_id':slide,'zoom':zoom,'magnification':zoom.replace('X',''),'dataset':dataset,'class_code':class_code,'subclass':subclass,'year':year,'sequence':sequence})
    except Exception as e:
        print(f'Error procesando {img_path}: {e}')
df=pd.DataFrame(registros)
totals={'total_imagenes':len(df),'total_pacientes':df["patient_id"].nunique(),'total_zooms':df["zoom"].nunique()}

# Analisis basico
stats_basicas={}
distribucion=df['label_name'].value_counts()
stats_df=pd.DataFrame({'Clase':distribucion.index,'Imagenes':distribucion.values,'Porcentaje':(distribucion.values/len(df)*100).round(2),'Pacientes Unicos':[df[df['label_name']==c]['patient_id'].nunique() for c in distribucion.index]})
ratio=stats_df['Imagenes'].max()/stats_df['Imagenes'].min() if len(stats_df['Imagenes'])>1 else 1
cv=stats_df['Imagenes'].std()/stats_df['Imagenes'].mean() if len(stats_df['Imagenes'])>0 else 0
stats_basicas={'total_imagenes':len(df),'total_pacientes':df["patient_id"].nunique(),'distribucion_clases':stats_df.to_dict('records'),'ratio_balanceo':ratio,'coeficiente_variacion':cv,'needs_balancing':ratio>1.5}
fig,axes=plt.subplots(1,3,figsize=(18,5))
bars=axes[0].bar(stats_df['Clase'],stats_df['Imagenes'],color=PALETTE)
axes[0].set_title('Distribucion por Clase',fontweight='bold',fontsize=12)
axes[0].set_xlabel('Clase',fontsize=10)
axes[0].set_ylabel('Numero de imagenes',fontsize=10)

for bar in bars:
    height=bar.get_height()
    axes[0].text(bar.get_x()+bar.get_width()/2,height+5,f'{int(height)}',ha='center',va='bottom',fontweight='bold',fontsize=9)

axes[1].pie(stats_df['Imagenes'],labels=stats_df['Clase'],autopct='%1.1f%%',startangle=90,colors=PALETTE)
axes[1].set_title('Porcentaje por Clase',fontweight='bold',fontsize=12)
axes[2].bar(stats_df['Clase'],stats_df['Pacientes Unicos'],color=PALETTE)
axes[2].set_title('Pacientes Unicos por Clase',fontweight='bold',fontsize=12)
axes[2].set_xlabel('Clase',fontsize=10)
axes[2].set_ylabel('Numero de pacientes',fontsize=10)

for i,val in enumerate(stats_df['Pacientes Unicos']):
    axes[2].text(i,val+0.5,str(val),ha='center',va='bottom',fontweight='bold',fontsize=9)
plt.tight_layout()
plt.show(block=True)

# Analisis por aumento (conteos y pacientes por zoom)
zoom_stats_df=df.groupby('zoom').agg({'label':'count','patient_id':'nunique','label_name':lambda x:(x=='benign').sum()}).reset_index()
zoom_stats_df.columns=['Zoom','Total_Imagenes','Pacientes_Unicos','Benignas']
zoom_stats_df['Malignas']=zoom_stats_df['Total_Imagenes']-zoom_stats_df['Benignas']
zoom_stats_df['Zoom']=pd.Categorical(zoom_stats_df['Zoom'],categories=ZOOM_ORDER,ordered=True)
zoom_stats_df=zoom_stats_df.sort_values('Zoom')
zoom_stats=zoom_stats_df.to_dict('records')
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].bar(zoom_stats_df['Zoom'],zoom_stats_df['Total_Imagenes'],color='skyblue',edgecolor='black')
axes[0].set_title('Total de imagenes por aumento',fontweight='bold',fontsize=12)
axes[0].set_xlabel('Aumento (zoom)',fontsize=10)
axes[0].set_ylabel('Numero de imagenes',fontsize=10)

for i,val in enumerate(zoom_stats_df['Total_Imagenes']):
    axes[0].text(i,val+5,str(val),ha='center',va='bottom',fontweight='bold',fontsize=9)

axes[1].bar(zoom_stats_df['Zoom'],zoom_stats_df['Pacientes_Unicos'],color='lightgreen',edgecolor='black')
axes[1].set_title('Pacientes unicos por aumento',fontweight='bold',fontsize=12)
axes[1].set_xlabel('Aumento (zoom)',fontsize=10)
axes[1].set_ylabel('Numero de pacientes',fontsize=10)

for i,val in enumerate(zoom_stats_df['Pacientes_Unicos']):
    axes[1].text(i,val+0.5,str(val),ha='center',va='bottom',fontweight='bold',fontsize=9)

plt.tight_layout()
plt.show(block=True)

# Analisis pacientes
imagenes_por_paciente=df.groupby('patient_id').size()
stats_pacientes={'media':imagenes_por_paciente.mean(),'mediana':imagenes_por_paciente.median(),'std':imagenes_por_paciente.std(),'min':imagenes_por_paciente.min(),'max':imagenes_por_paciente.max(),'q1':imagenes_por_paciente.quantile(0.25),'q3':imagenes_por_paciente.quantile(0.75),'pacientes_con_1_imagen':(imagenes_por_paciente==1).sum(),'pacientes_con_mas_de_10_imagenes':(imagenes_por_paciente>10).sum(),'total_pacientes':len(imagenes_por_paciente),'coeficiente_variacion':imagenes_por_paciente.std()/imagenes_por_paciente.mean()}
pacientes_solo_benignos=df.groupby('patient_id').filter(lambda x:(x['label_name']=='benign').all())
pacientes_solo_malignos=df.groupby('patient_id').filter(lambda x:(x['label_name']=='malignant').all())
pacientes_mixtos=df.groupby('patient_id').filter(lambda x:len(x['label_name'].unique())>1)
num_mixtos=pacientes_mixtos['patient_id'].nunique()
print(f'Pacientes mixtos (ambas clases): {num_mixtos}')
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].hist(imagenes_por_paciente,bins=30,edgecolor='black',alpha=0.7,color='steelblue')
axes[0].axvline(stats_pacientes['media'],color='red',linestyle='--',linewidth=2,label=f"Media:{stats_pacientes['media']:.1f}")
axes[0].axvline(stats_pacientes['mediana'],color='green',linestyle='--',linewidth=2,label=f"Mediana:{stats_pacientes['mediana']:.1f}")
axes[0].set_title('Distribucion de imagenes por paciente (histograma)',fontweight='bold',fontsize=12)
axes[0].set_xlabel('Imagenes por paciente',fontsize=10)
axes[0].set_ylabel('Numero de pacientes',fontsize=10)
axes[0].legend()
axes[1].boxplot(imagenes_por_paciente,vert=True,patch_artist=True,boxprops=dict(facecolor='lightblue'),medianprops=dict(color='red',linewidth=2))
axes[1].set_title('Dispersion de imagenes por paciente (boxplot)',fontweight='bold',fontsize=12)
axes[1].set_ylabel('Imagenes por paciente',fontsize=10)
axes[1].set_xticklabels(['Pacientes'])
plt.tight_layout()
plt.show(block=True)
tipos=['Solo benignos','Solo malignos','Mixtos']
counts=[pacientes_solo_benignos['patient_id'].nunique(),pacientes_solo_malignos['patient_id'].nunique(),num_mixtos]
patient_stats={'imagenes_por_paciente':stats_pacientes,'tipos_pacientes':dict(zip(tipos,counts)),'pacientes_mixtos':num_mixtos}

# Analisis imagenes muestra (muestra balanceada por clase)
sample_df=df.groupby('label_name',group_keys=False).apply(lambda x:x.sample(min(len(x),sample_images//2),random_state=42)).sample(frac=1,random_state=42)
resultados={'resoluciones':[],'blur_scores':[],'contraste_scores':[],'brillo_promedio':[],'entropia':[],'zoom':[],'label':[],'patient_id':[]}
for _,row in tqdm(sample_df.iterrows(),total=len(sample_df),desc='Procesando imagenes',disable=True):
    try:
        img_path=row['filepath']
        img=Image.open(img_path)
        img_array=np.array(img)
        resultados['resoluciones'].append(img.size)
        gray=cv2.cvtColor(img_array,cv2.COLOR_RGB2GRAY) if len(img_array.shape)==3 else img_array
        resultados['blur_scores'].append(cv2.Laplacian(gray,cv2.CV_64F).var())
        resultados['contraste_scores'].append(np.std(gray))
        resultados['brillo_promedio'].append(np.mean(gray))
        hist=cv2.calcHist([gray],[0],None,[256],[0,256])
        hist=hist/hist.sum()
        resultados['entropia'].append(-np.sum(hist*np.log2(hist+1e-10)))
        resultados['zoom'].append(row['zoom'])
        resultados['label'].append(row['label_name'])
        resultados['patient_id'].append(row['patient_id'])
        img.close()
    except Exception as e:
        print(f"Error procesando {row['filename']}: {e}")
df_resultados=pd.DataFrame(resultados)
image_stats=df_resultados
numeric_cols=['blur_scores','contraste_scores','brillo_promedio','entropia']
metrics_info=[('blur_scores','Nitidez (varianza Laplaciana)','Puntuacion de blur'),('contraste_scores','Contraste','Desviacion estandar de intensidad'),('brillo_promedio','Brillo','Intensidad promedio'),('entropia','Entropia','Entropia (bits)')]
fig_class,axes_class=plt.subplots(2,2,figsize=(14,10))

for ax,(col,title,ylabel) in zip(axes_class.flatten(),metrics_info):
    sns.histplot(data=df_resultados,x=col,hue='label',element='step',stat='density',common_norm=False,fill=True,alpha=0.4,ax=ax)
    ax.set_title(f'{title} por clase',fontweight='bold')
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Densidad')

plt.tight_layout()
plt.show(block=True)
fig_zoom,axes_zoom=plt.subplots(2,2,figsize=(14,10))

for ax,(col,title,ylabel) in zip(axes_zoom.flatten(),metrics_info):
    sns.histplot(data=df_resultados,x=col,hue='zoom',element='step',stat='density',common_norm=False,fill=True,alpha=0.35,ax=ax)
    ax.set_title(f'{title} por aumento',fontweight='bold')
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Densidad')

plt.tight_layout()
plt.show(block=True)

if all(col in df_resultados.columns for col in numeric_cols):
    fig_corr,ax_corr=plt.subplots(figsize=(6,5))
    corr_matrix=df_resultados[numeric_cols].corr()
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0,square=True,ax=ax_corr,fmt='.2f',cbar_kws={'label':'Coeficiente de correlacion'})
    ax_corr.set_title('Correlacion de metricas de imagen',fontweight='bold')
    ax_corr.set_xlabel('Metrica')
    ax_corr.set_ylabel('Metrica')
    plt.tight_layout()
    plt.show(block=True)

# Visualizaciones de imagenes por subtipo (muestra compacta en matriz) y matriz aumentos-subtipos
if 'subclass' in df.columns and not df['subclass'].isnull().all():
    subtypes=df['subclass'].unique().tolist()
    samples=[]
    for sub in subtypes:
        subset=df[df['subclass']==sub].sample(min(2,len(df[df['subclass']==sub])),random_state=42)
        for _,row in subset.iterrows():
            samples.append((row['filepath'],row['label_name'],sub))
    if samples:
        ncols=4
        nrows=int(np.ceil(len(samples)/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows))
        axes=np.atleast_1d(axes).flatten()
        for i,ax in enumerate(axes):
            if i >= len(samples):
                ax.axis('off')
                continue
            img_path,label_name,sub=samples[i]
            img=Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{label_name} - {sub}",fontsize=9)
            img.close()
        plt.tight_layout()
        plt.show(block=True)

    # Matriz 4x4: 4 subtipos aleatorios x 4 aumentos
    available_subs=subtypes.copy()
    random.shuffle(available_subs)
    selected_subs=available_subs[:4] if len(available_subs)>=4 else available_subs
    available_zooms=df['zoom'].unique().tolist()
    if len(available_zooms)>4:
        random.shuffle(available_zooms)
        selected_zooms=available_zooms[:4]
    else:
        selected_zooms=available_zooms
    fig_grid,axes_grid=plt.subplots(len(selected_zooms),len(selected_subs),figsize=(3*len(selected_subs),3*len(selected_zooms)))
    for i,zm in enumerate(selected_zooms):
        for j,sub in enumerate(selected_subs):
            ax=axes_grid[i,j] if len(selected_zooms)>1 else axes_grid[j]
            subset=df[(df['zoom']==zm)&(df['subclass']==sub)]
            if len(subset)==0:
                ax.axis('off')
                ax.set_title(f'{zm} - {sub}\n(no img)',fontsize=8)
                continue
            row=subset.sample(1,random_state=42).iloc[0]
            img=Image.open(row['filepath'])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{zm} - {sub}',fontsize=8)
            img.close()
    plt.tight_layout()
    plt.show(block=True)

# Analisis subclases (barras por subclase y pacientes)
subclase_stats=None
if 'subclass' in df.columns:
    subclases_counts=df['subclass'].value_counts()
    subclases_por_clase=pd.crosstab(df['subclass'],df['label_name'])
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    bars1=axes[0].bar(range(len(subclases_counts)),subclases_counts.values,color='lightcoral',edgecolor='black')
    axes[0].set_xticks(range(len(subclases_counts)))
    axes[0].set_xticklabels(subclases_counts.index,rotation=45,ha='right')
    axes[0].set_title('Distribucion de subclases',fontweight='bold')
    axes[0].set_xlabel('Subclase')
    axes[0].set_ylabel('Numero de imagenes')

    for bar in bars1:
        height=bar.get_height()
        axes[0].text(bar.get_x()+bar.get_width()/2,height+5,f'{int(height)}',ha='center',va='bottom',fontsize=8,fontweight='bold')
    
    subclases_orden=subclases_counts.index
    if 'benign' in subclases_por_clase.columns and 'malignant' in subclases_por_clase.columns:
        benign_counts=subclases_por_clase.loc[subclases_orden,'benign']
        malign_counts=subclases_por_clase.loc[subclases_orden,'malignant']
        axes[1].bar(range(len(subclases_orden)),benign_counts,label='Benign',color=PALETTE[0],edgecolor='black')
        axes[1].bar(range(len(subclases_orden)),malign_counts,bottom=benign_counts,label='Malignant',color=PALETTE[1],edgecolor='black')
        axes[1].legend()
        axes[1].set_xticks(range(len(subclases_orden)))
        axes[1].set_xticklabels(subclases_orden,rotation=45,ha='right')
        axes[1].set_title('Composicion por clase dentro de cada subclase',fontweight='bold')
        axes[1].set_xlabel('Subclase')
        axes[1].set_ylabel('Numero de imagenes')

    plt.tight_layout()
    plt.show(block=True)
    pacientes_por_subclase=df.groupby('subclass')['patient_id'].nunique()
    fig2,ax=plt.subplots(figsize=(12,6))
    bars=ax.bar(range(len(pacientes_por_subclase)),pacientes_por_subclase.values,color='lightgreen',edgecolor='black')
    ax.set_xticks(range(len(pacientes_por_subclase)))
    ax.set_xticklabels(pacientes_por_subclase.index,rotation=45,ha='right')
    ax.set_title('Pacientes unicos por subclase',fontweight='bold')
    ax.set_xlabel('Subclase')
    ax.set_ylabel('Numero de pacientes')

    for bar,val in zip(bars,pacientes_por_subclase.values):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,str(val),ha='center',va='bottom',fontsize=9,fontweight='bold')
    
    plt.tight_layout()
    plt.show(block=True)
    subclase_stats={'subclases':subclases_counts.to_dict(),'subclases_por_clase':subclases_por_clase.to_dict(),'pacientes_por_subclase':pacientes_por_subclase.to_dict()}

# Analisis split por paciente
pacientes_info=[]

for patient_id in df['patient_id'].unique():
    patient_data=df[df['patient_id']==patient_id]
    class_type='mixed' if patient_data['label_name'].nunique()>1 else patient_data['label_name'].iloc[0]
    info={'patient_id':patient_id,'total_images':len(patient_data),'benign_count':(patient_data['label_name']=='benign').sum(),'malignant_count':(patient_data['label_name']=='malignant').sum(),'unique_classes':patient_data['label_name'].nunique(),'class_type':class_type,'zooms':', '.join(patient_data['zoom'].unique())}
    pacientes_info.append(info)

pacientes_df=pd.DataFrame(pacientes_info)
resumen={'pacientes_1_imagen':int((pacientes_df['total_images']==1).sum()),'pacientes_2_5':int(((pacientes_df['total_images']>=2)&(pacientes_df['total_images']<=5)).sum()),'pacientes_6_10':int(((pacientes_df['total_images']>=6)&(pacientes_df['total_images']<=10)).sum()),'pacientes_mas_10':int((pacientes_df['total_images']>10).sum()),'pacientes_puros':int((pacientes_df['unique_classes']==1).sum()),'pacientes_mixtos':int((pacientes_df['unique_classes']>1).sum())}
split_info={'pacientes_df':pacientes_df.to_dict('records'),'resumen':resumen}

# Analisis calidad (blur y contraste basicos)
sample=df.sample(min(len(df),200),random_state=42)
blur_scores=[]
contrast_scores=[]

for _,row in tqdm(sample.iterrows(),total=len(sample),desc='Calidad',disable=True):
    try:
        img=Image.open(row['filepath'])
        arr=np.array(img)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY) if len(arr.shape)==3 else arr
        blur_scores.append(cv2.Laplacian(gray,cv2.CV_64F).var())
        contrast_scores.append(np.std(gray))
        img.close()
    except Exception:
        continue

blur_scores=pd.Series(blur_scores)
contrast_scores=pd.Series(contrast_scores)
problemas={'alto_blur':float((blur_scores<100).mean()*100),'bajo_contraste':float((contrast_scores<30).mean()*100)}
calidad_stats={'blur_media':float(blur_scores.mean()),'contraste_media':float(contrast_scores.mean()),'problemas_porcentaje':problemas}
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].hist(blur_scores,bins=30,color='skyblue',edgecolor='black')
ax[0].axvline(100,color='red',linestyle='--',label='Umbral blur')
ax[0].legend()
ax[0].set_title('Distribucion del blur',fontweight='bold')
ax[0].set_xlabel('Puntuacion de blur (varianza Laplaciana)')
ax[0].set_ylabel('Frecuencia')
ax[1].hist(contrast_scores,bins=30,color='lightgreen',edgecolor='black')
ax[1].axvline(30,color='red',linestyle='--',label='Umbral contraste')
ax[1].legend()
ax[1].set_title('Distribucion del contraste',fontweight='bold')
ax[1].set_xlabel('Desviacion estandar de intensidad')
ax[1].set_ylabel('Frecuencia')
plt.tight_layout()
plt.show(block=True)

# Analisis color y texturas (promedios por canal e indicadores GLCM)
sample=df.sample(min(len(df),200),random_state=42)
color_records=[]
glcm_records=[]
for _,row in tqdm(sample.iterrows(),total=len(sample),desc='Color',disable=True):
    try:
        img_arr=np.array(Image.open(row['filepath']).convert('RGB'))
        ch_means=img_arr.reshape(-1,3).mean(axis=0)
        color_records.append({'label':row['label_name'],'zoom':row['zoom'],'R':float(ch_means[0]),'G':float(ch_means[1]),'B':float(ch_means[2])})
        if SKIMAGE_AVAILABLE:
            gray=cv2.cvtColor(img_arr,cv2.COLOR_RGB2GRAY)
            gray=cv2.resize(gray,(128,128),interpolation=cv2.INTER_AREA)
            quant=(gray/8).astype(np.uint8)
            glcm=greycomatrix(quant,distances=[1],angles=[0,np.pi/4,np.pi/2,3*np.pi/4],levels=32,symmetric=True,normed=True)
            glcm_records.append({'label':row['label_name'],'contrast':float(greycoprops(glcm,'contrast').mean()),'homogeneity':float(greycoprops(glcm,'homogeneity').mean()),'energy':float(greycoprops(glcm,'energy').mean()),'entropy':float(-np.sum(glcm*np.log2(glcm+1e-10)))})
    except Exception:
        continue

stats_color={}

if color_records:
    df_color=pd.DataFrame(color_records)
    df_color_long=df_color.melt(id_vars=['label','zoom'],value_vars=['R','G','B'],var_name='Canal',value_name='Intensidad')
    fig_z,axes_z=plt.subplots(1,3,figsize=(18,5))
    for ax,canal in zip(axes_z,['R','G','B']):
        sns.histplot(data=df_color_long[df_color_long['Canal']==canal],x='Intensidad',hue='zoom',element='step',common_norm=False,stat='density',fill=True,alpha=0.35,ax=ax)
        ax.set_title(f'Distribucion {canal} por aumento',fontweight='bold')
        ax.set_xlabel('Intensidad media por imagen')
        ax.set_ylabel('Densidad')
    plt.tight_layout()
    plt.show(block=True)
    stats_color['por_clase']=_agg_color(df_color,'label')
    stats_color['por_zoom']=_agg_color(df_color,'zoom')

texture_stats={}

if SKIMAGE_AVAILABLE and glcm_records:
    glcm_df=pd.DataFrame(glcm_records)
    metrics=['contrast','homogeneity','energy','entropy']
    fig_glcm_c,axes_glcm_c=plt.subplots(2,2,figsize=(14,10))

    for ax,metric in zip(axes_glcm_c.flatten(),metrics):
        sns.barplot(data=glcm_df,x='label',y=metric,ax=ax,edgecolor='black')
        ax.set_title(f'{metric.capitalize()} GLCM por clase',fontweight='bold')
        ax.set_xlabel('Clase')
        ax.set_ylabel(metric.capitalize())

    plt.tight_layout()
    plt.show(block=True)
    texture_stats['por_clase']={lbl:{m:float(sub[m].mean()) for m in metrics} for lbl,sub in glcm_df.groupby('label')}

elif not SKIMAGE_AVAILABLE:
    print('scikit-image no disponible: se omite analisis GLCM.')

color_texture_stats={'color':stats_color,'textura':texture_stats}

print('Resumen ejecucion')
print(f"- Imagenes: {totals.get('total_imagenes',0)}")
print(f"- Pacientes: {totals.get('total_pacientes',0)}")
print(f"- Zooms: {totals.get('total_zooms',0)}")
if stats_basicas:
    print(f"- Balance imgs: {stats_basicas.get('ratio_balanceo',0):.2f}:1")
    print(f"- Varianza imgs (coef var): {stats_basicas.get('coeficiente_variacion',0):.3f}")
if patient_stats:
    mixtos=patient_stats.get('pacientes_mixtos',0)
    total=totals.get('total_pacientes',1)
    print(f"- Pacientes mixtos: {mixtos} ({mixtos/total*100:.1f}%)")
