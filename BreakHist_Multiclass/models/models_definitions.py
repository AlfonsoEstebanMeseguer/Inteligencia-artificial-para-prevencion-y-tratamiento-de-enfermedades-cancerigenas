import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# CNN 1 Multiclase (parecida a CNN1 Binaria)
def build_cnn_light_1(input_shape,num_classes):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(32,(3,3),padding="same")(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(64,(3,3),padding="same")(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(128,(3,3),padding="same")(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x) #softmax para multiclase
    return models.Model(inputs,outputs)

# CNN2 Multiclase (parecida a CNN2 Binaria (VGG-like))
def build_cnn_light_2(input_shape,num_classes,l2_reg,dropout):
    inputs=layers.Input(shape=input_shape)
    x=layers.Conv2D(32,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(32,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(64,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(64,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(128,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(128,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(256,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.Conv2D(256,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
    x=layers.ReLU()(layers.BatchNormalization()(x))
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    return models.Model(inputs,outputs)

# CNN Transfer Multiclase (basada en EfficientNetB0, con fine-tuning a partir de la capa 200)
def build_cnn_transfer_1(input_shape, num_classes,dropout,trainable_at):
    base=tf.keras.applications.EfficientNetB0(include_top=False,weights="imagenet",input_shape=input_shape)
    # Configuramos qué capas serán entrenables en fine-tuning; capa inicial desactivada, parte final activada.
    for idx,layer in enumerate(base.layers):
        layer.trainable=idx>=trainable_at # desde donde marcamos como entrenable hasta el final, el resto congelado
    base.trainable=False
    inp=tf.keras.Input(shape=input_shape)
    x=base(inp,training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(dropout)(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(dropout)(x)
    out=layers.Dense(num_classes,activation="softmax")(x)
    model=tf.keras.Model(inp,out)
    return model,base
