import tensorflow as tf
from typing import Dict, Tuple

# Niveles de augmentación
NONE="none"
LOW="low"
MEDIUM="medium"
ADVANCED="advanced"
EXPERT="expert"

# Diccionarios de reutilización
ROT_LAYERS={}
ZOOM_LAYERS={}
TRANS_LAYERS={}

params={
    NONE:{"rotation":0.0,"zoom":0.0,"shift":0.0,"brightness":0.0,"contrast":0.0,"flip_prob":0.0,"hue":0.0,"saturation":0.0,"cutout_prob":0.0},
    LOW:{"rotation":5.0,"zoom":0.05,"shift":0.05,"brightness":0.05,"contrast":0.05,"flip_prob":0.3,"hue":0.02,"saturation":0.05,"cutout_prob":0.0},
    MEDIUM:{"rotation":15.0,"zoom":0.1,"shift":0.1,"brightness":0.1,"contrast":0.1,"flip_prob":0.5,"hue":0.05, "saturation":0.1, "cutout_prob":0.1},
    ADVANCED:{"rotation":25.0, "zoom":0.2,"shift":0.15,"brightness":0.15,"contrast":0.15,"flip_prob":0.7,"hue":0.1,"saturation":0.15,"cutout_prob":0.2,"mixup_alpha":0.2,"cutmix_alpha":1.0},
    EXPERT:{"rotation":30.0,"zoom":0.25,"shift":0.2,"brightness":0.2,"contrast":0.2,"flip_prob":0.8,"hue":0.08,"saturation":0.12,"cutout_prob":0.25,"mixup_alpha": 0.4,"cutmix_alpha":1.0,"gaussian_noise":0.05,"speckle_noise":0.02},
}

"""
Función que devuelve un booleano aleatorio con probabilidad 'prob' de ser True o false. si prob=0.3 -> ~=30% de veces True. Esta función nos servirá para
aplicar ciertas augmentaciones de forma aleatoria como fliops o cutout. Podríamos siempre usar flips para augment pero es mejor no hacerlo siempre para asegurar
diversidad en los datos aumentados. 

Usamos tf.random.uniform para que sea compatible con tf.data.Dataset.map. Dentyro de tf.data no se pueden usar funciones de random de numpy o python puro como random.random()
(no es tensorizable). 
"""
def random_bool(prob):
    random_value=tf.random.uniform([],0.0,1.0)
    if random_value < prob:
        return True
    else:
        return False

"""
Las siguientes funciones han sido implementadas con el fin de que las capas de augmentación de Keras no creen variables nuevas (con mismo valor) cada vez que se llaman 
dentro de una función decorada con @tf.function (como es el caso de tf.data.Dataset.map). Esto se consigue almacenando las capas ya creadas en un diccionario
y reutilizándolas cuando se solicitan con los mismos parámetros. De esta forma, evitamos la creación de variables dentro de tf.function (reduciendo el tiempo de ejecución
y evitando errores).
"""
def get_rot_layer(rotation_deg):
    key=float(rotation_deg)
    layer=ROT_LAYERS.get(key)
    if layer is None:
        layer=tf.keras.layers.RandomRotation(key/360.0)
        ROT_LAYERS[key]=layer
    return layer

def get_zoom_layer(zoom_val):
    key=float(zoom_val)
    layer=ZOOM_LAYERS.get(key)
    if layer is None:
        layer=tf.keras.layers.RandomZoom(key,key)
        ZOOM_LAYERS[key]=layer
    return layer

def get_shift_layer(shift_val):
    key=float(shift_val)
    layer=TRANS_LAYERS.get(key)
    if layer is None:
        layer=tf.keras.layers.RandomTranslation(key,key)
        TRANS_LAYERS[key]=layer
    return layer

"""
Función que aplica augmentaciones básicas a una imagen según los parámetros dados. Estas augmentaciones incluyen rotación, zoom, desplazamiento, 
flips, brillo, contraste, tono y saturación.
"""
def apply_basic_augmentations(image,params,img_size):
    if params.get("rotation",0) > 0:
        image = get_rot_layer(params["rotation"])(image,training=True)
    if params.get("zoom",0) > 0:
        image = get_zoom_layer(params["zoom"])(image,training=True)
    if params.get("shift",0) > 0: 
        image = get_shift_layer(params["shift"])(image,training=True)
    if params.get("flip_prob",0) > 0:
        if random_bool(params.get("flip_prob",0)):
            image=tf.image.flip_left_right(image)
        if random_bool(params.get("flip_prob",0)/2):
            image=tf.image.flip_up_down(image)

    if params.get("brightness",0) > 0:
        image=tf.image.random_brightness(image,params["brightness"])
    if params.get("contrast",0) > 0:
        """
        Hacemos contraste aleatorio marcado con dos límites, ejemplo con contrast = 0.1
        1-0.1 = 0.9 -> reducimos contraste hasta un mínimo (10%)
        1+0.1 = 1.1 -> el contraste máximo agregado es un (10%)
        básicamente máximo rango es -10%->+10% de contraste
        """
        image=tf.image.random_contrast(image,1-params["contrast"],1+params["contrast"])
    if params.get("hue",0) > 0:
        # Hue es la tonalidad, buscamos elevar un poco regiones más coloridas o que no aprenda por "sello" de color por el instrumento usado para las imágenes del dataset.
        image=tf.image.random_hue(image,params["hue"])
    if params.get("saturation",0) > 0: # misma estratregia que random_contrast
        image=tf.image.random_saturation(image,1-params["saturation"],1+params["saturation"])

    """
    Augmentación extra: recorte aleatorio del 90% de la imagen y redimensionado a tamaño original. Esto puede ayudar a que el modelo no se fije demasiado en los bordes de las imágenes y
    aprenda a generalizar mejor. Solo se aplica si la imagen es mayor de 100x100 píxeles (aunque el dataset asegura dimensiones 700x460 -> queremos evitar fugas)
    para evitar perder demasiada información en imágenes pequeñas. Nota: si usamos esto debe ser tras resize porque se trabaja sobre el tensor de la imagen actual. 
    Diferencia de crop con zoom: el zoom puede dejar bordes vacíos (rellenados con 0s) si se hace zoom out, mientras que el crop siempre recorta una parte central y redimensiona.
    """
    if img_size[0] > 100 and img_size[1] > 100:
        crop_h=max(int(img_size[0]*0.9),1)  # recorte del 90% de la altura, mínimo 1 píxel
        crop_w=max(int(img_size[1]*0.9),1)  # recorte del 90% de la anchura, mínimo 1 píxel
        image=tf.image.random_crop(image,[crop_h,crop_w,3])  # definimos el recorte en la imagen (tensor)
        image=tf.image.resize(image,img_size) # redimensionamos a tamaño original

    return image

"""
Función que aplica cutout a una imagen. La técnica cutout es otra de las muchas técnicas de augmentación. Básicamente lo que hace es crear un "parche" (o cuadrado/rectángulo) que 
sea de un mismo color y que tape algo de la imagen. Como buscamos que el modelo generalice lo máximo posible, no queremos que se agarre a nada, por eso le tapamos regiones aleatorias 
para que pueda aprender de más patrones. PERO OJO -> esta técnica es peligrosa porque podrías estar eliminando contínuamente zonas interesantes para aprender del tejido cancerígeno. 
Por eso solo se aplica si se cumple una probabilidad definida en los parámetros de configuración de augmentación y en una zona aleatoria de la imagen que ocupa el 20% (por defecto)
"""
def apply_cutout(image,img_size, ratio=0.2):
    h,w =img_size
    # Aunque las imágenes supuestamente sean todas tamañó 700x460, vamos a definir tamaño de parche dependiendo del tamaño de la imagen: decisión de diseño
    #tamaño
    cut_h=int(h*ratio)
    cut_w=int(w*ratio)
    # posiciones
    y=tf.random.uniform([],0,h-cut_h,dtype=tf.int32)
    x=tf.random.uniform([],0,w-cut_w,dtype=tf.int32)
    # definimos puntos de parche
    # Esto es una máscara que se va a superponer sobre la imagen original, lo que buscamos hacer para provocar el cutout es:
    # pixel * mascara de ese pixel, si píxel * 1 = pixel, pero si píxel * 0 = entonces negro
    mask=tf.ones([h,w,3],dtype=image.dtype) 
    # región a poner a 0
    zeros=tf.zeros([cut_h,cut_w,3],dtype=image.dtype) # tamaño parche
    paddings=[[y,h-y-cut_h],[x,w-x-cut_w],[0,0]] #0,0 porque no hacemos padding a canales
    # lógica: coordenada de parche inicial + tamaño parche + resto = total -> h (eje de coorenada 'y', para 'x' sería 'w')
    # Entonces nuestro objetivo es definir que el parche llegará desde y(punto donde empieza) hasta el punto donde termina,
    # despejando la fórmula tenemos que = resto=total−coordenada de parche inicial−tamaño parche
    # Colocamos el parche en (x,y) (que se expanda al tamaño que pusimos a partir de (x,y))
    cutout_mask=tf.pad(zeros,paddings,constant_values=1.0) # Definimos la máscara de 0 sobre (x,y)
    mask=mask*cutout_mask # Máscara inicial con máscara de píxeles 0 encima
    return image*mask # Aplicamos máscara 

"""
Función principal que aplica las augmentaciones a una imagen según el nivel y parámetros definidos en la configuración.
"""
def apply_augmentations(image:tf.Tensor,config):
    
    level=config["augmentation_level"]
    params_cfg=config["augmentation_params"] 
    img_size=config["img_size"] 

    if level==NONE:
        return image
    # Aseguramos que los valores de la imagen estén en el rango [0,1]. Esto ya se ha explicado, pero yo primero augmento y luego normalizo, así es 
    # la práctica adecuada para este proceso. Por ello, las imágenes previamente a ser normalizadas, poseen píxeles con valores 0-255, si aplico
    # augmentaciones sobre esos valores (por ejemplo, brillo), entonces se dispara el efecto. Debemos normalizar para obtener augmentaciones también sobre 0-1
    image=tf.clip_by_value(image,0.0,1.0)
    # Aplicamos siempre augmentaciones suaves
    image=apply_basic_augmentations(image,params_cfg,img_size)
    # SOLO si augmentaciones avanzadas:
    if level in (ADVANCED, EXPERT):
        if params_cfg.get("cutout_prob",0) > 0 and random_bool(params_cfg["cutout_prob"]):
            # Aplicamos cutout solo si se cumple probabilidad; ratio fijo (20%) salvo que se añada en params
            cut_ratio=params_cfg.get("cutout_ratio",0.2)
            image=apply_cutout(image,img_size,ratio=cut_ratio)
        # NIVEL EXPERTO: OJO!!!! -> EXPERIMENTAL
        if level==EXPERT:
            if params_cfg.get("gaussian_noise",0) > 0:
                # Añadimos ruido aditivo -> riesgo porque el color sí tiene significado, el ruido PODRÍA distorsionar un poco el color y crear patrones irreales, 
                # Creo que puede tener peores resultados en histología -> solo mi suposición, por eso experimental
                noise=tf.random.normal(tf.shape(image),0.0,params_cfg["gaussian_noise"])
                image=tf.clip_by_value(image+noise,0.0,1.0)
            if params_cfg.get("speckle_noise",0)> 0:
                # Añadimos ruido multiplicativo -> 
                noise=tf.random.normal(tf.shape(image),0.0,params_cfg["speckle_noise"])
                image=tf.clip_by_value(image+image*noise,0.0,1.0)

    return image
