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
# MODELO CNN LIGERO
# =========================================================

def build_cnn_light(
    input_shape=(224, 224, 3),
    num_classes=1,
    l2_reg=1e-4,
):
    """
    CNN ligera entrenada desde cero para clasificación binaria.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Bloque 1
    x = layers.Conv2D(
        32, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloque 2
    x = layers.Conv2D(
        64, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloque 3
    x = layers.Conv2D(
        128, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="last_conv",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
    model = models.Model(inputs, outputs, name="HistologyCNN_Light")
    return model


# =========================================================
# UTILIDADES
# =========================================================

def ensure_splits(base_path, split_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    """Crea los JSON de splits si no existen."""
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
        path = os.path.join(split_dir, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits[split], f, indent=2)


def get_datasets(config, split_dir):
    """Crea datasets tf.data listos para model.fit con steps calculados."""
    loader = BreakHisDataLoader(config)

    train_imgs, train_labels = loader.load_split(split_dir, DataSplit.TRAIN)
    val_imgs, val_labels = loader.load_split(split_dir, DataSplit.VALIDATION)
    test_imgs, test_labels = loader.load_split(split_dir, DataSplit.TEST)

    if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
        raise RuntimeError("Algún split está vacío. Revisa los JSON de splits.")

    train_ds = loader.create_dataset(train_imgs, train_labels, training=True)
    val_ds = loader.create_dataset(val_imgs, val_labels, training=False)
    test_ds = loader.create_dataset(test_imgs, test_labels, training=False)

    steps_per_epoch = math.ceil(len(train_imgs) / config.batch_size)
    val_steps = math.ceil(len(val_imgs) / config.batch_size)
    test_steps = math.ceil(len(test_imgs) / config.batch_size)

    class_weights = loader.compute_class_weights(train_labels) if config.use_class_weights else None

    return {
        "loader": loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "test_steps": test_steps,
        "class_weights": class_weights,
        "test_imgs": test_imgs,
        "test_labels": test_labels,
    }


def evaluate_model(model, test_ds, test_steps, threshold=0.5):
    """Evalúa en test y devuelve métricas y predicciones."""
    y_true = []
    y_prob = []

    for batch_imgs, batch_labels in test_ds.take(test_steps):
        preds = model.predict(batch_imgs, verbose=0)
        y_true.append(batch_labels.numpy().ravel())
        y_prob.append(preds.ravel())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["benign", "malignant"], output_dict=True)

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
    return metrics, cm, y_true, y_prob, y_pred


def plot_confusion(cm, save_path="confusion_matrix.png", labels=("Benign", "Malignant")):
    """Guarda matriz de confusión."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Genera heatmap Grad-CAM para una imagen preprocesada."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_example(model, loader, image_path, output_path, last_conv_layer_name="last_conv"):
    """Guarda un overlay Grad-CAM para una imagen dada."""
    # Imagen original sin normalizar para overlay
    raw_bytes = tf.io.read_file(image_path)
    orig = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    orig = tf.image.resize(orig, loader.config.img_size).numpy().astype(np.uint8)

    # Imagen preprocesada (igual que dataset)
    img = loader.decode_image(tf.constant(image_path))
    img, _ = loader.preprocess_image(img, tf.constant(0), training=False)
    img_array = tf.expand_dims(img, axis=0)

    heatmap = make_gradcam_heatmap(model, img_array, last_conv_layer_name)
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, loader.config.img_size).numpy().astype(np.uint8)

    overlay = 0.4 * heatmap[..., 0:1] + orig
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap[..., 0], cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =========================================================
# MAIN
# =========================================================

def main():
    BASE_PATH = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"
    SPLITS_DIR = "splits"
    EPOCHS = 40
    LR = 1e-3

    # Configuración del dataset
    config = DatasetConfig(
        img_size=(224, 224),
        batch_size=32,
        augmentation_level=AugmentationLevel.MEDIUM,
        normalization_mode=NormalizationMode.IMAGENET,
        use_class_weights=True,
    )

    ensure_splits(BASE_PATH, SPLITS_DIR, train_size=0.8, val_size=0.1, test_size=0.1)
    ds_bundle = get_datasets(config, SPLITS_DIR)

    model = build_cnn_light(input_shape=(*config.img_size, 3), num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),
            tf.keras.metrics.SensitivityAtSpecificity(0.9, name="sens_at_spec90"),
            tf.keras.metrics.SpecificityAtSensitivity(0.9, name="spec_at_sens90"),
        ],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "cnn_light_best.h5", monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        ds_bundle["train_ds"],
        validation_data=ds_bundle["val_ds"],
        epochs=EPOCHS,
        steps_per_epoch=ds_bundle["steps_per_epoch"],
        validation_steps=ds_bundle["val_steps"],
        class_weight=ds_bundle["class_weights"],
        callbacks=callbacks,
        verbose=1,
    )

    print("\n🧪 Evaluación en TEST:")
    model.evaluate(ds_bundle["test_ds"], steps=ds_bundle["test_steps"], verbose=2)

    metrics, cm, y_true, y_prob, y_pred = evaluate_model(
        model, ds_bundle["test_ds"], ds_bundle["test_steps"], threshold=0.5
    )
    print("\n📊 Métricas finales:", json.dumps(metrics, indent=2))
    print("\n📉 Classification report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))

    cm_path = plot_confusion(cm, save_path="confusion_matrix.png")
    print(f"📌 Matriz de confusión guardada en: {cm_path}")

    # Grad-CAM en algunas imágenes de test
    sample_paths = ds_bundle["test_imgs"][:3]
    gradcam_dir = "gradcam_examples"
    os.makedirs(gradcam_dir, exist_ok=True)
    for i, img_path in enumerate(sample_paths, 1):
        out_path = os.path.join(gradcam_dir, f"gradcam_{i}.png")
        try:
            save_gradcam_example(
                model,
                ds_bundle["loader"],
                img_path,
                out_path,
                last_conv_layer_name="last_conv",
            )
            print(f"🖼️ Grad-CAM guardado en: {out_path}")
        except Exception as e:
            print(f"⚠️ No se pudo generar Grad-CAM para {img_path}: {e}")

    model.save("cnn_light_breakhis.h5")
    print("💾 Modelo guardado en cnn_light_breakhis.h5")


if __name__ == "__main__":
    main()
