import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# CNN #1 – LIGHT
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
def build_cnn_light(input_shape=(224,224,3),num_classes=1,l2_reg=1e-4):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(32,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(64,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(128,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg),name="last_conv")(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(num_classes,activation="sigmoid")(x)
    model=models.Model(inputs,outputs)
    return model

# CNN #2 – VGG-LIKE
"""
Para este segundo modelo se han definido dos funciones, la primera es simplemente una función que construye un bloque convolucional basado en vgg, es decir:
varios bloques convolucionales antes de reducir la resolución y posteriormente maxpooling. La justficcación de VGG antes que una CNN simple como la del modelo 1 es la siguiente:
Maxpooling reduce la información a consta de obtener más patrones por menos parámetros, es necesario PERO DESTRUCTIVO. En una cnn simple al principio se detectan patrones muy sencillos, que 
rápidamente son reducidos, muchos detalles no llegan a las capas profundas ante tanta destrucción (patrones quizás necesarios para la precisión que requiere la histología médica). 
VGG fue un diseño que buscaba solventar esto -> más patrones antes de reducir, primera conv detecta patrones básicos, segunda conv combina esos patrones y ahora SI reducimos (información
más rica). Otra ventaja es que es el resultado de aplicar filtros más complejos con la ventaja de que la convolución en secuencia de filtros más pequeños es más rápida que un filtro grande,
pero con el mismo resultado. Propiedad "CASCADING"

La función "build_cnn_vgg_histology" simplemente se aprovecha de conv_block para la construcción de una cnn
"""
def conv_block(x,filters,l2_reg,name):
    x=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg),name=f"{name}_conv1")(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg),name=f"{name}_conv2")(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    return x

def build_cnn_vgg_histology(input_shape=(224,224,3),num_classes=1,l2_reg=1e-4):
    inputs=layers.Input(shape=input_shape)
    x=conv_block(inputs,32,l2_reg,"block1")
    x=conv_block(x,64,l2_reg,"block2")
    x=conv_block(x,128,l2_reg,"block3")
    x=conv_block(x,256,l2_reg,"block4")
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256, activation="relu")(x)
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(1, activation="sigmoid")(x)
    model=models.Model(inputs,outputs)
    return model


# CNN #3 – MULTI-SCALE (INCEPTION-LIKE)
def inception_block(x,filters,l2_reg,name):
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
def build_cnn3_multiscale(input_shape=(224,224,3),num_classes=1,l2_reg=1e-4):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(32,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=inception_block(x,32,l2_reg,"inc1")
    x=layers.MaxPooling2D((2,2))(x)
    x=inception_block(x,64,l2_reg,"inc2")
    x=layers.MaxPooling2D((2,2))(x)
    x=inception_block(x,96,l2_reg,"inc3")
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(128,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg),name="last_conv")(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(1,activation="sigmoid")(x)
    model=models.Model(inputs,outputs)
    return model


# CNN #4 – RESIDUAL (RESNET-18 LIKE)
def residual_block(x,filters,stride,l2_reg,name):
    shortcut=x
    y=layers.Conv2D(filters,(3,3),strides=stride,padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(x)
    y=layers.BatchNormalization()(y)
    y=layers.ReLU()(y)
    y=layers.Conv2D(filters,(3,3),strides=1,padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(y)
    y=layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut=layers.Conv2D(filters,(1,1),strides=stride,padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(shortcut)
        shortcut=layers.BatchNormalization()(shortcut)
    out=layers.Add()([y, shortcut])
    out=layers.ReLU()(out)
    return out

"""
Esta función agrupa bloques residuales. Primero reducimos resolución a cambio de duplicar canales de filtros, este primer bloque marcara estos parámetros.
En los siguientes (dentro del bucle), mantendrán la resolución y canales pero profundizan mucho en el aprendizaje aprendido. La motivación de esto es
aprender MÁS transformaciones al MISMO NIVEL. 
"""
def make_stage(x,filters,blocks,first_stride,l2_reg,stage_name):
    x=residual_block(x,filters,stride=first_stride,l2_reg=l2_reg,name=f"{stage_name}_block1")
    for i in range(1,blocks):
        x=residual_block(x,filters,stride=1,l2_reg=l2_reg,name=f"{stage_name}_block{i+1}")
    return x

"""
Finalmente construimos el modelo final, empezando como una cnn simple (capas stem). Ahora declaro 8 bloques residuales (4*2), refinamos las 
características con la última convolución (512), agrupamos en una capa para aplanar parámetros y usamos DOS DROPOUTS. Porque? pues porque 
en la primera capa densa (512) las características extraidas son muchas pero poco definidas, no queremos que se sesge demasiado. Por eso hacemos el primer dropout.
Ahora  hacemos otra capa densa(256) para combinar las evidencias simples anteriores, esta si que posee evidencias más complejas. Pero como tampoco queremos que se sesge y
depende demasiado de patrones complejos, volvemos a atacar con otro dropout (DOBLE CONTROL ANTIOVERFITTING)
"""
def build_cnn4_residual(input_shape=(224,224,3),num_classes=1,l2_reg=1e-4,dropout=0.4):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(64,(3,3),strides=2,padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
    x=layers.MaxPooling2D((3,3),strides=2,padding="same")(x)
    x=make_stage(x,64,blocks=2,first_stride=1,l2_reg=l2_reg,stage_name="stage1")
    x=make_stage(x,128,blocks=2,first_stride=2,l2_reg=l2_reg,stage_name="stage2")
    x=make_stage(x,256,blocks=2,first_stride=2,l2_reg=l2_reg,stage_name="stage3")
    x=make_stage(x,512,blocks=2,first_stride=2,l2_reg=l2_reg,stage_name="stage4")
    x=layers.Conv2D(512,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg),name="last_conv_raw")(x)
    x=layers.BatchNormalization(name="last_conv_bn")(x)
    x=layers.ReLU(name="last_conv")(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(dropout)(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(1, activation="sigmoid")(x)
    model=models.Model(inputs,outputs)
    return model

# CNN #5 – DENSENET + SE
"""
Bloque SE: piensa en un mando de volumen por canal. Primero miramos cada canal de forma global (promedio de toda la imagen),
luego dos densas deciden qué canales subir o bajar (sigmoide 0-1). Con esos pesos reescalamos el tensor original.
Traducción a humano: si un canal aporta, lo iluminamos; si no, lo atenuamos, sin cambiar la forma de la imagen.
"""
def se_block(x,reduction=16,name="se"):
    c=x.shape[-1]
    s=layers.GlobalAveragePooling2D()(x)
    s=layers.Dense(max(c//reduction,8),activation="relu")(s)
    s=layers.Dense(c,activation="sigmoid")(s)
    s=layers.Reshape((1,1,c))(s)
    return layers.Multiply()([x, s])
"""
Capa densa de DenseNet: primero BN+ReLU para dejar todo limpio/positivo. Un 1x1 comprime canales (cálculo barato),
un 3x3 extrae patrones nuevos, y al final concatenamos la salida con la entrada. Así cada capa conserva todo lo anterior
y vamos “apilando” canales (growth_rate) como una bola de nieve de features.
"""
def dense_layer(x,growth_rate,l2_reg,drop_rate,name):
    y=layers.ReLU()(layers.BatchNormalization()(x))
    y=layers.Conv2D(4*growth_rate,(1,1),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(y)
    y=layers.ReLU()(layers.BatchNormalization()(y))
    y=layers.Conv2D(growth_rate,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(y)
    if drop_rate and drop_rate>0:
        y=layers.Dropout(drop_rate)(y)
    x=layers.Concatenate()([x,y])
    return x

"""
Bloque denso: varias dense_layer conectadas por concatenación (todas ven todo lo previo) y, al final, un SE que decide qué canales son más o menos útiles.
Resultado: muchos detalles guardados y un filtro de prioridad al terminar el bloque.
"""
def dense_block(x,n_layers,growth_rate,l2_reg,drop_rate,name):
    for i in range(1,n_layers+1):
        x=dense_layer(x,growth_rate,l2_reg,drop_rate,name=f"{name}_l{i}")
    x=se_block(x,reduction=16,name=f"{name}_se")
    return x

"""
Transición: frenamos el crecimiento. Un 1x1 reduce canales según compression y un average pooling baja la resolución.
Es como comprimir equipaje antes de seguir metiendo ropa: mantienes lo importante, ocupas menos y puedes añadir más bloques.
"""
def transition_layer(x,compression,l2_reg,drop_rate,name):
    n_channels=int(x.shape[-1]*compression)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(n_channels,(1,1),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(x)
    if drop_rate and drop_rate>0:
        x=layers.Dropout(drop_rate)(x)
    x=layers.AveragePooling2D((2,2),strides=2,padding="same")(x)
    return x

"""
DenseNet-SE completa: arrancamos con un stem sencillo, metemos bloques densos (que guardan todo lo aprendido) separados por transiciones que compactan,
y al final GAP + cabeza densa con dropout para la decisión binaria. Cada bloque lleva SE para auto-regular qué canales importan en histología.
"""
def build_cnn5_densenet_se(input_shape=(224,224,3),growth_rate=24,block_layers=(6,10,14),compression=0.5,l2_reg=1e-4,drop_rate=0.2):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(64,(3,3),strides=2,padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((3,3),strides=2,padding="same")(x)
    x=dense_block(x,block_layers[0],growth_rate,l2_reg,drop_rate,name="db1")
    x=transition_layer(x,compression,l2_reg,drop_rate,name="tr1")
    x=dense_block(x,block_layers[1],growth_rate,l2_reg,drop_rate,name="db2")
    x=transition_layer(x,compression,l2_reg,drop_rate,name="tr2")
    x=dense_block(x,block_layers[2],growth_rate,l2_reg,drop_rate,name="db3")
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(256,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(l2_reg),name="last_conv_raw")(x)
    x=layers.BatchNormalization(name="last_conv_bn")(x)
    x=layers.ReLU(name="last_conv")(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.4)(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(0.4)(x)
    outputs=layers.Dense(1,activation="sigmoid")(x)
    return models.Model(inputs, outputs)

# TRANSFER LEARNING MODELS
"""
ResNet50 transfer: usamos la ResNet50 de ImageNet como “ojos” ya entrenados y la congelamos para no destrozar sus pesos.
GAP para aplanar y una cabeza pequeñita (Dense + dos dropouts) que decide benigno/maligno. Idea: reutilizar un buen extractor y solo ajustar lo mínimo, con regularización extra.
"""
def build_resnet50_transfer(input_shape=(224,224,3),dropout=0.4):
    inp=tf.keras.Input(shape=input_shape)
    base=tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_tensor=inp)
    base.trainable=False
    x=base.output
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(dropout)(x)
    x=tf.keras.layers.Dense(256,activation="relu")(x)
    x=tf.keras.layers.Dropout(dropout)(x)
    out=tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model=tf.keras.Model(inp,out)
    return model,base

"""
# EfficientNetB0 transfer: mismo concepto. EfficientNetB0 de ImageNet congelada, GAP y cabeza binaria con dropout.
# EfficientNet es compacta y eficiente; solo tocamos la parte final para nuestro problema y evitamos recalentar todo el backbone.
"""
def build_efficientnetb0_transfer(input_shape=(224,224,3),dropout=0.4):
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
