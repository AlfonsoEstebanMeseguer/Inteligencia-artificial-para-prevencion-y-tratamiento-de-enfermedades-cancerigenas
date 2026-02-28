import tensorflow as tf
from tensorflow.keras import layers,models,regularizers
from tensorflow.keras.applications import EfficientNetB2

"""
Este modelo es el mismo que el de clasificación binaria, pero con la cabeza adaptada a multiclase (softmax y más neuronas), 
además de que se le da la opción de elegir hasta qué capa entrenar del EfficientNetB2
"""
def build_bracs_cnn_transfer_1(input_shape,num_classes,dropout,trainable_at):
    base=EfficientNetB2(include_top=False,weights="imagenet",input_shape=input_shape)
    # rescaling en h5 necesita valores numpy, si no peta
    for layer in base.layers:
        if isinstance(layer,tf.keras.layers.Rescaling):
            if hasattr(layer.scale,"numpy"):
                layer.scale=layer.scale.numpy()
            if hasattr(layer.offset,"numpy"):
                layer.offset=layer.offset.numpy()
    idx=0
    for layer in base.layers: # marcamos hasta donde entrenar
        if idx>=trainable_at:
            layer.trainable=True
        else:
            layer.trainable=False
        idx+=1
    base.trainable=False
    inputs=layers.Input(shape=input_shape)
    x=base(inputs,training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(dropout)(x)
    x=layers.Dense(512,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    model=models.Model(inputs,outputs,name="bracs_cnn_transfer_1")
    return model,base
