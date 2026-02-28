import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# CNN #1 (para modelos 1)
"""
Este modelo es el más básico de los que usaremos en la clasificación binaria:
Primero defino el tensor de entrada (224x244x3 canales). Ahora sí, empezamos con las capas ocultas de convolución.
1º Bloque de convolución: no hacemos padding(buscamos tamñaños iguales, no eliminar bordes, por conveniencia y por mayor compatibilidad en operaciones: 
la reducción de dimensiones solo ocurriría en el pooling)
adicionalmente usamos regularicación L2 para penalizar pesos grandes, muy usada para reducir sobreentrenamiento en datasets. Para mejorar el aprendizaje,
tras la convolución obtenemos tensores con valores muy dispares, vamos a suavizar los gradientes con batchNormalization. Después, con valores más escalados 
aplicamos Relu para eliminar negativos, pues necesitamos evitar ambiguedades en las sumas de las convoluciones. Para patrones no detectados, los pondremos a 0,
asi la suma ponderada de cada capa de convolución solo será de patrones detectados. Permitir negativos puede hacer que patrones detectados se anulen al hacer:
peso * valor

2º y 3º Bloques siguen la misma lógica
Por último convertimos las capas 28×28×128 en un solo vector de 128 -> equivalente a capa flatten. No usamos flatten porque la multiplicación resultante daría muchos
parámetros (aumentando un posible riesgo de overfitting) y carga de trabajo. Por su contraparte, gap (golbalaveragepooling2D) no necesita almacenar tanta información

Para terminar defino una capa de neuronas densa que conecta cada neurona con cada entrada del vector anteriormente delcarado y las aplasto todas en una capa de neuronas
sigmoid (salida binaria, num clases=1(0/1)). 

Nota: usamos dropout para mejorar la generalización, forzamos a que no aprenda de memoria con demasiadas neuronas mirando
Nota2: Se obviaran explicaciones anteriores en futuros comentarios
"""
def build_cnn_light(input_shape,l2_reg,dropout):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(4,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(8,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(16,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(16,activation="relu")(x)
    x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(1,activation="sigmoid")(x)
    model=models.Model(inputs,outputs)
    return model

# CNN MULTI-SCALE 
def inception_block(x,filters,l2_reg):
    b1=layers.Conv2D(filters,(1,1),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    b1=layers.ReLU()(layers.BatchNormalization()(b1))
    b2=layers.Conv2D(filters,(1,1),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    b2=layers.ReLU()(layers.BatchNormalization()(b2))
    b2=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(b2)
    b2=layers.ReLU()(layers.BatchNormalization()(b2))
    b3=layers.Conv2D(filters,(1,1),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    b3=layers.ReLU()(layers.BatchNormalization()(b3))
    b3=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(b3)
    b3=layers.ReLU()(layers.BatchNormalization()(b3))
    b3=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(b3)
    b3=layers.ReLU()(layers.BatchNormalization()(b3))
    b4=layers.MaxPooling2D((3,3),strides=(1,1),padding="same")(x)
    b4=layers.Conv2D(filters,(1,1),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(b4)
    b4=layers.ReLU()(layers.BatchNormalization()(b4))
    x=layers.Concatenate()([b1,b2,b3,b4])
    x=layers.Conv2D(filters*2,(1,1),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    return x

"""
Empezamos como una cnn normal, detectando bordes y reduciendo tamaño. A continuación introducimos los bloques de multiramas para aumentar la capacidad y generalización.
Terminamos como una cnn normal (3x3 -> BN-> RELU) y aplanamos pasra pasar a una capa densa sigmoid para clasificación binaria.
"""
def build_cnn3_multiscale(input_shape,l2_reg,dropout):
    head_units=128
    stem_filters=24
    inc_filters=(24,48,72)
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(stem_filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=inception_block(x,inc_filters[0],l2_reg)
    x=layers.MaxPooling2D((2,2))(x)
    x=inception_block(x,inc_filters[1],l2_reg)
    x=layers.MaxPooling2D((2,2))(x)
    x=inception_block(x,inc_filters[2],l2_reg)
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(max(64,inc_filters[2]),(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(head_units,activation="relu")(x)
    x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(1,activation="sigmoid")(x)
    model=models.Model(inputs,outputs)
    return model

"""
EfficientNetB0 transfer: EfficientNetB0 de ImageNet congelada, GAP y cabeza binaria con dropout.
"""
def build_efficientnetb0_transfer(input_shape,dropout):
    inp=tf.keras.Input(shape=input_shape)
    base=tf.keras.applications.EfficientNetB0(include_top=False,weights="imagenet",input_tensor=inp)
    base.trainable=False
    x=base.output
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(dropout)(x)
    x=tf.keras.layers.Dense(256,activation="relu")(x)
    x=tf.keras.layers.Dropout(dropout)(x)
    out=tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model=tf.keras.Model(inp,out)
    return model,base
