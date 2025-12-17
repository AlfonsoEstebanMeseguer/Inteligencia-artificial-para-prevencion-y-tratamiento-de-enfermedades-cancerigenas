import os
import math
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
# MODELO CNN #2 – VGG-LIKE HISTOLOGY
# =========================================================

def conv_block(x, filters, l2_reg, name):
    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(x)
    return x


def build_cnn_vgg_histology(
    input_shape=(224, 224, 3),
    num_classes=1,
    l2_reg=1e-4,
):
    """
    CNN #2 – VGG-like para histología (desde cero)
    Mayor capacidad que CNN #1.
    """

    inputs = layers.Input(shape=input_shape, name="input_image")

    x = conv_block(inputs, 32, l2_reg, "block1")
    x = conv_block(x, 64, l2_reg, "block2")
    x = conv_block(x, 128, l2_reg, "block3")
    x = conv_block(x, 256, l2_reg, "block4")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = models.Model(inputs, outputs, name="HistologyCNN_VGG")
    return model


# =========================================================
# UTILIDADES (idénticas a CNN #1)
# =========================================================

def ensure_splits(base_path, split_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    if all(os.path.exists(os.path.join(split_dir, f"{s}.json")) for s in ["train", "val", "test"]):
        return

    os.makedirs(split_dir, exist_ok=True)
    _, all_images, all_labels, _, slides = read_binary_breakhis_data(base_path, verbose=False)

    splits, _ = split_by_patient(
        all_images,
        all_labels,
        slides,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    for split in ["train", "val", "test"]:
        with open(os.path.join(split_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(splits[split], f, indent=2)


def get_datasets(config, split_dir):
    loader = BreakHisDataLoader(config)

    train_imgs, train_labels = loader.load_split(split_dir, DataSplit.TRAIN)
    val_imgs, val_labels = loader.load_split(split_dir, DataSplit.VALIDATION)
    test_imgs, test_labels = loader.load_split(split_dir, DataSplit.TEST)

    train_ds = loader.create_dataset(train_imgs, train_labels, training=True)
    val_ds = loader.create_dataset(val_imgs, val_labels, training=False)
    test_ds = loader.create_dataset(test_imgs, test_labels, training=False)

    steps = lambda x: math.ceil(len(x) / config.batch_size)

    class_weights = loader.compute_class_weights(train_labels) if config.use_class_weights else None

    return {
        "loader": loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "steps_train": steps(train_imgs),
        "steps_val": steps(val_imgs),
        "steps_test": steps(test_imgs),
        "class_weights": class_weights,
        "test_imgs": test_imgs,
        "test_labels": test_labels,
    }


def evaluate_model(model, test_ds, test_steps, threshold=0.5):
    y_true, y_prob = [], []

    for imgs, labels in test_ds.take(test_steps):
        preds = model.predict(imgs, verbose=0)
        y_true.append(labels.numpy().ravel())
        y_prob.append(preds.ravel())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["benign", "malignant"],
        output_dict=True
    )

    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "specificity": cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8),
        "sensitivity": cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-8),
    }
    return metrics, cm, y_true, y_pred


# =========================================================
# MAIN
# =========================================================

def main():
    BASE_PATH = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"
    SPLITS_DIR = "splits"
    EPOCHS = 50
    LR = 1e-3

    config = DatasetConfig(
        img_size=(224, 224),
        batch_size=32,
        augmentation_level=AugmentationLevel.MEDIUM,
        normalization_mode=NormalizationMode.IMAGENET,
        use_class_weights=True,
    )

    ensure_splits(BASE_PATH, SPLITS_DIR)
    ds = get_datasets(config, SPLITS_DIR)

    model = build_cnn_vgg_histology(
        input_shape=(*config.img_size, 3),
        num_classes=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
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
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "cnn_vgg_histology_best.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    model.fit(
        ds["train_ds"],
        validation_data=ds["val_ds"],
        epochs=EPOCHS,
        steps_per_epoch=ds["steps_train"],
        validation_steps=ds["steps_val"],
        class_weight=ds["class_weights"],
        callbacks=callbacks,
        verbose=1,
    )

    print("\n🧪 Evaluación en TEST:")
    model.evaluate(ds["test_ds"], steps=ds["steps_test"], verbose=2)

    metrics, cm, y_true, y_pred = evaluate_model(
        model, ds["test_ds"], ds["steps_test"]
    )

    print("\n📊 Métricas finales:")
    print(json.dumps(metrics, indent=2))

    print("\n📉 Classification report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))

    model.save("cnn_vgg_histology_final.h5")
    print("💾 Modelo guardado")


if __name__ == "__main__":
    main()
