import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_light_1(input_shape=(224,224,3), num_classes=4):
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
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    return models.Model(inputs,outputs)

def build_cnn_light_2(input_shape=(224,224,3), num_classes=4, l2_reg=1e-4, dropout=0.4):
    def conv_block(x,filters):
        x=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
        x=layers.ReLU()(layers.BatchNormalization()(x))
        x=layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(l2_reg))(x)
        x=layers.ReLU()(layers.BatchNormalization()(x))
        x=layers.MaxPooling2D((2,2))(x)
        return x

    inputs=layers.Input(shape=input_shape)
    x=conv_block(inputs,32)
    x=conv_block(x,64)
    x=conv_block(x,128)
    x=conv_block(x,256)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    return models.Model(inputs,outputs)


def build_cnn_transfer_1(input_shape=(224,224,3), num_classes=4, dropout=0.4, trainable_at=200):
    base=tf.keras.applications.EfficientNetB0(include_top=False,weights="imagenet",input_shape=input_shape)
    # Configuramos qué capas serán entrenables en fine-tuning; capa inicial desactivada, parte final activada.
    for idx,layer in enumerate(base.layers):
        layer.trainable=idx>=trainable_at
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
