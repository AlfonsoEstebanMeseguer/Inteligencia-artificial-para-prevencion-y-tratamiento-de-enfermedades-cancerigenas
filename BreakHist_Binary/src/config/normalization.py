import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Modos de normalización
IMAGENET="imagenet"
RESNET="resnet"
EFFICIENTNET="efficientnet"
CUSTOM="custom"
STANDARD="standard"  
"""
Función que normaliza la imagen para cada tipo de preprocesado. Recibe la config completa y usa
el campo normalization_mode.
"""
def normalize_image(image,config):
    mode=config.get("normalization_mode",IMAGENET)
    image=tf.cast(image,tf.float32)
    if mode==IMAGENET:  
        """
        El método de normalización de imagenet normaliza a 0-1(diviendo por 255) y luego resta las medias de cada canal y divide
        por sus desviaciones estándar. Así cada canal queda centrado y escalado según la distribución de Imagenet. Esto se hace así
        porque la mayoría de arquitecturas preentrenadas esperan entradas con esas estadísticas. Los valores son fijos y son
        calculados con las millones de imágenes que ya ha visto imagenet para entrenar, los datos de estos valores son públicos y extraíbles
        """
        image=image/255.0
        result=(image-[0.485,0.456,0.406])/[0.229,0.224,0.225]
        return result
    if mode==RESNET:
        return resnet_preprocess(image)
    if mode==EFFICIENTNET:
        return efficientnet_preprocess(image)
    if mode in (STANDARD,CUSTOM):
        return image/255.0
    raise ValueError(f"Modo de normalización no soportado:{mode}")
