import os
import math
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from config_dataset import (
    DatasetConfig,
    AugmentationLevel,
    NormalizationMode,
    DataSplit,
    BreakHisDataLoader,
)
from readDataset import read_binary_breakhis_data
from split_dataset import split_by_patient


# =========================================================
# UTILIDADES (splits + datasets + evaluación)
# =========================================================

def ensure_splits(base_path, split_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    if all(os.path.exists(os.path.join(split_dir, f"{s}.json")) for s in ["train", "val", "test"]):
        return

    os.makedirs(split_dir, exist_ok=True)
    _, all_images, all_labels, _, slides = read_binary_breakhis_data(base_path, verbose=False)

    splits, _ = split_by_patient(
        all_images, all_labels, slides,
        train_size=train_size, val_size=val_size, test_size=test_size
    )

    for split in ["train", "val", "test"]:
        with open(os.path.join(split_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(splits[split], f, indent=2)


def get_datasets(config, split_dir):
    loader = BreakHisDataLoader(config)

    train_imgs, train_labels = loader.load_split(split_dir, DataSplit.TRAIN)
    val_imgs, val_labels     = loader.load_split(split_dir, DataSplit.VALIDATION)
    test_imgs, test_labels   = loader.load_split(split_dir, DataSplit.TEST)

    train_ds = loader.create_dataset(train_imgs, train_labels, training=True)
    val_ds   = loader.create_dataset(val_imgs, val_labels, training=False)
    test_ds  = loader.create_dataset(test_imgs, test_labels, training=False)

    steps = lambda x: math.ceil(len(x) / config.batch_size)

    # Para esta CNN #5, usaremos focal loss con alpha derivado del desbalance.
    # Aun así devolvemos class_weights por si quieres comparar (pero por defecto no las usamos).
    class_weights = loader.compute_class_weights(train_labels) if config.use_class_weights else None

    return {
        "loader": loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "steps_train": steps(train_imgs),
        "steps_val": steps(val_imgs),
        "steps_test": steps(test_imgs),
        "train_labels": np.array(train_labels, dtype=np.int32),
        "val_labels": np.array(val_labels, dtype=np.int32),
        "test_labels": np.array(test_labels, dtype=np.int32),
        "class_weights": class_weights,
        "test_imgs": test_imgs,
        "val_imgs": val_imgs,
    }


def predict_probs(model, ds, steps):
    y_prob = []
    y_true = []
    for x, y in ds.take(steps):
        p = model.predict(x, verbose=0).ravel()
        y_prob.append(p)
        y_true.append(y.numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def metrics_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=["benign", "malignant"],
        output_dict=True,
        zero_division=0
    )

    out = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "specificity": cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8),
        "sensitivity": cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-8),
    }
    return out, cm, y_pred


def tune_threshold_on_val(y_true, y_prob):
    """
    Busca el umbral que maximiza Youden's J = sensitivity + specificity - 1
    (muy típico en medicina). Alternativa: max F1.
    """
    best_t = 0.5
    best_j = -1e9

    for t in np.linspace(0.05, 0.95, 91):
        _, cm, _ = metrics_from_probs(y_true, y_prob, threshold=t)
        spec = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
        sens = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-8)
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)

    return best_t, best_j


# =========================================================
# LOSS: FOCAL + LABEL SMOOTHING (para evitar colapso)
# =========================================================

class BinaryFocalLossWithSmoothing(tf.keras.losses.Loss):
    """
    Focal loss binaria con alpha (balanceo) + gamma (dureza) + label smoothing.
    Muy útil en datasets desbalanceados.
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.05, name="binary_focal_smooth"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        # label smoothing: y = y*(1-s) + 0.5*s
        if self.label_smoothing and self.label_smoothing > 0:
            s = self.label_smoothing
            y_true = y_true * (1.0 - s) + 0.5 * s

        # focal loss
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        loss = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)


# =========================================================
# CNN #5 – DenseNet-BC + SE Attention (from scratch)
# =========================================================

def se_block(x, reduction=16, name="se"):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    s = layers.Dense(max(c // reduction, 8), activation="relu", name=f"{name}_fc1")(s)
    s = layers.Dense(c, activation="sigmoid", name=f"{name}_fc2")(s)
    s = layers.Reshape((1, 1, c), name=f"{name}_reshape")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])


def dense_layer(x, growth_rate, l2_reg, drop_rate, name):
    # Bottleneck (1x1) + 3x3 (DenseNet-BC)
    y = layers.BatchNormalization(name=f"{name}_bn1")(x)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(
        4 * growth_rate, (1, 1), padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv1x1"
    )(y)

    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.ReLU(name=f"{name}_relu2")(y)
    y = layers.Conv2D(
        growth_rate, (3, 3), padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv3x3"
    )(y)

    if drop_rate and drop_rate > 0:
        y = layers.Dropout(drop_rate, name=f"{name}_drop")(y)

    x = layers.Concatenate(name=f"{name}_concat")([x, y])
    return x


def dense_block(x, n_layers, growth_rate, l2_reg, drop_rate, name):
    for i in range(1, n_layers + 1):
        x = dense_layer(x, growth_rate, l2_reg, drop_rate, name=f"{name}_l{i}")
    x = se_block(x, reduction=16, name=f"{name}_se")  # SE tras el bloque (mejora canal)
    return x


def transition_layer(x, compression, l2_reg, drop_rate, name):
    # Reduce canales y hace downsample
    n_channels = int(x.shape[-1] * compression)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        n_channels, (1, 1), padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv1x1"
    )(x)
    if drop_rate and drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"{name}_drop")(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding="same", name=f"{name}_avgpool")(x)
    return x


def build_cnn5_densenet_se(
    input_shape=(224, 224, 3),
    growth_rate=24,
    block_layers=(6, 10, 14),   # equilibrado: potente pero no descomunal desde cero
    compression=0.5,
    l2_reg=1e-4,
    drop_rate=0.2,
):
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem: 3x3 para texturas finas (histología)
    x = layers.Conv2D(
        64, (3, 3), strides=2, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name="stem_pool")(x)

    # Dense blocks
    x = dense_block(x, block_layers[0], growth_rate, l2_reg, drop_rate, name="db1")
    x = transition_layer(x, compression, l2_reg, drop_rate, name="tr1")

    x = dense_block(x, block_layers[1], growth_rate, l2_reg, drop_rate, name="db2")
    x = transition_layer(x, compression, l2_reg, drop_rate, name="tr2")

    x = dense_block(x, block_layers[2], growth_rate, l2_reg, drop_rate, name="db3")

    # last_conv para Grad-CAM
    x = layers.BatchNormalization(name="last_bn")(x)
    x = layers.ReLU(name="last_relu")(x)
    x = layers.Conv2D(
        256, (3, 3), padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="last_conv"
    )(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="head_drop")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="head_drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    return models.Model(inputs, outputs, name="HistologyCNN_DenseNetSE")


# =========================================================
# MAIN
# =========================================================

def main():
    BASE_PATH = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"
    SPLITS_DIR = "splits"

    # Más épocas: esta red suele necesitar más, pero early stopping corta.
    EPOCHS = 120
    BATCH = 32

    # Config: subimos augment a ADVANCED para mejorar generalización
    config = DatasetConfig(
        img_size=(224, 224),
        batch_size=BATCH,
        augmentation_level=AugmentationLevel.ADVANCED,  # más fuerte que CNN3/CNN4
        normalization_mode=NormalizationMode.IMAGENET,
        use_class_weights=False,  # aquí usaremos focal loss (balanceo interno)
    )

    ensure_splits(BASE_PATH, SPLITS_DIR)
    ds = get_datasets(config, SPLITS_DIR)

    # Alpha para focal: aproximación basada en proporciones de train
    # y=1 (malignant) suele ser mayoritario -> alpha para clase positiva más bajo.
    pos = ds["train_labels"].mean()  # proporción de 1s
    # Queremos penalizar más la minoritaria (benign=0), por eso alpha ~ pos (al revés).
    # En focal binaria, alpha se aplica a y=1, así que ponemos alpha = 1-pos.
    alpha = float(1.0 - pos)
    alpha = min(max(alpha, 0.10), 0.90)

    loss_fn = BinaryFocalLossWithSmoothing(alpha=alpha, gamma=2.0, label_smoothing=0.05)

    model = build_cnn5_densenet_se(
        input_shape=(*config.img_size, 3),
        growth_rate=24,
        block_layers=(6, 10, 14),
        compression=0.5,
        l2_reg=1e-4,
        drop_rate=0.2,
    )

    # Optimizador: Adam con Cosine Decay (TF 2.10 no soporta weight_decay en Adam)
    steps_per_epoch = ds["steps_train"]
    total_steps = steps_per_epoch * EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=total_steps
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("cnn5_densenet_se_best.h5", monitor="val_auc", mode="max",
                                           save_best_only=True, verbose=1),
    ]

    history = model.fit(
        ds["train_ds"],
        validation_data=ds["val_ds"],
        epochs=EPOCHS,
        steps_per_epoch=ds["steps_train"],
        validation_steps=ds["steps_val"],
        callbacks=callbacks,
        verbose=1,
    )

    # =====================================================
    # 1) Threshold tuning en VALIDACIÓN (médico, válido)
    # =====================================================
    yv_true, yv_prob = predict_probs(model, ds["val_ds"], ds["steps_val"])
    best_t, best_j = tune_threshold_on_val(yv_true, yv_prob)

    print(f"\n🎯 Threshold óptimo (val) por Youden J: t={best_t:.2f}  (J={best_j:.3f})")

    # =====================================================
    # 2) Evaluación en TEST (0.50 y threshold óptimo)
    # =====================================================
    yt_true, yt_prob = predict_probs(model, ds["test_ds"], ds["steps_test"])

    print("\n🧪 TEST @ threshold=0.50")
    m50, cm50, yp50 = metrics_from_probs(yt_true, yt_prob, threshold=0.50)
    print(json.dumps(m50, indent=2))
    print(classification_report(yt_true, yp50, target_names=["benign", "malignant"], zero_division=0))

    print(f"\n🧪 TEST @ threshold óptimo val={best_t:.2f}")
    mt, cmt, ypt = metrics_from_probs(yt_true, yt_prob, threshold=best_t)
    print(json.dumps(mt, indent=2))
    print(classification_report(yt_true, ypt, target_names=["benign", "malignant"], zero_division=0))

    model.save("cnn5_densenet_se_final.h5")
    print("\n💾 Modelo guardado: cnn5_densenet_se_final.h5")


if __name__ == "__main__":
    main()
