import os, math, json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score

from config_dataset import (
    DatasetConfig, AugmentationLevel, NormalizationMode, DataSplit, BreakHisDataLoader
)
from readDataset import read_binary_breakhis_data
from split_dataset import split_by_patient


def ensure_splits(base_path, split_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    if all(os.path.exists(os.path.join(split_dir, f"{s}.json")) for s in ["train", "val", "test"]):
        return
    os.makedirs(split_dir, exist_ok=True)
    _, all_images, all_labels, _, slides = read_binary_breakhis_data(base_path, verbose=False)
    splits, _ = split_by_patient(all_images, all_labels, slides, train_size=train_size, val_size=val_size, test_size=test_size)
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

    steps = lambda n: math.ceil(n / config.batch_size)
    class_weights = loader.compute_class_weights(train_labels) if config.use_class_weights else None

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "steps_train": steps(len(train_imgs)),
        "steps_val": steps(len(val_imgs)),
        "steps_test": steps(len(test_imgs)),
        "class_weights": class_weights,
    }

def predict_probs(model, ds, steps):
    y_true, y_prob = [], []
    for x, y in ds.take(steps):
        p = model.predict(x, verbose=0).ravel()
        y_true.append(y.numpy().ravel())
        y_prob.append(p)
    return np.concatenate(y_true), np.concatenate(y_prob)

def metrics_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=["benign", "malignant"], output_dict=True, zero_division=0)
    out = {
        "accuracy": rep["accuracy"],
        "precision": rep["weighted avg"]["precision"],
        "recall": rep["weighted avg"]["recall"],
        "f1": rep["weighted avg"]["f1-score"],
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "specificity": cm[0,0] / (cm[0,0] + cm[0,1] + 1e-8),
        "sensitivity": cm[1,1] / (cm[1,1] + cm[1,0] + 1e-8),
    }
    return out, cm, y_pred

def tune_threshold_youden(y_true, y_prob):
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 91):
        _, cm, _ = metrics_from_probs(y_true, y_prob, threshold=float(t))
        spec = cm[0,0] / (cm[0,0] + cm[0,1] + 1e-8)
        sens = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-8)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def build_efficientnetb0_transfer(input_shape=(224,224,3), dropout=0.4):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inp = tf.keras.Input(shape=input_shape, name="input_image")
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop2")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = tf.keras.Model(inp, out, name="BreakHis_EfficientNetB0_TL")
    return model, base


def main():
    BASE_PATH  = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"
    SPLITS_DIR = "splits"

    # Para EfficientNet: tu normalizador usa preprocess_input(efficientnet) y espera [0,255] :contentReference[oaicite:2]{index=2}
    config = DatasetConfig(
        img_size=(224,224),
        batch_size=32,
        augmentation_level=AugmentationLevel.ADVANCED,
        normalization_mode=NormalizationMode.EFFICIENTNET,
        use_class_weights=True,
        cache=True
    )

    ensure_splits(BASE_PATH, SPLITS_DIR)
    ds = get_datasets(config, SPLITS_DIR)

    model, base = build_efficientnetb0_transfer(input_shape=(*config.img_size, 3), dropout=0.4)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

    # --- Fase 1: cabeza ---
    opt1 = tfa.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0)
    model.compile(
        optimizer=opt1,
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            "cnn7_effnetb0_best.weights.h5",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
    ]

    model.fit(
        ds["train_ds"],
        validation_data=ds["val_ds"],
        epochs=15,
        steps_per_epoch=ds["steps_train"],
        validation_steps=ds["steps_val"],
        class_weight=ds["class_weights"],
        callbacks=cbs,
        verbose=1,
    )

    # --- Fase 2: fine-tuning parcial ---
    base.trainable = True
    # EfficientNet suele ir mejor afinando un poco más de arriba, pero sin abrirla entera
    for layer in base.layers[:-40]:
        layer.trainable = False

    opt2 = tfa.optimizers.AdamW(learning_rate=2e-5, weight_decay=1e-4, clipnorm=1.0)
    model.compile(
        optimizer=opt2,
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    cbs2 = [
        tf.keras.callbacks.ModelCheckpoint(
            "cnn7_effnetb0_finetuned_best.weights.h5",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3, patience=4, min_lr=1e-7, verbose=1),
    ]

    model.fit(
        ds["train_ds"],
        validation_data=ds["val_ds"],
        epochs=40,
        steps_per_epoch=ds["steps_train"],
        validation_steps=ds["steps_val"],
        class_weight=ds["class_weights"],
        callbacks=cbs2,
        verbose=1,
    )

    # --- Threshold tuning en VAL ---
    yv_true, yv_prob = predict_probs(model, ds["val_ds"], ds["steps_val"])
    t_best, j_best = tune_threshold_youden(yv_true, yv_prob)
    print(f"\n🎯 Threshold óptimo (VAL) Youden J: t={t_best:.2f} (J={j_best:.3f})")

    # --- TEST report (t=0.50 y t=t_best) ---
    yt_true, yt_prob = predict_probs(model, ds["test_ds"], ds["steps_test"])

    print("\n🧪 TEST @ threshold=0.50")
    m50, _, yp50 = metrics_from_probs(yt_true, yt_prob, threshold=0.50)
    print(json.dumps(m50, indent=2))
    print(classification_report(yt_true, yp50, target_names=["benign", "malignant"], zero_division=0))

    print(f"\n🧪 TEST @ threshold óptimo VAL={t_best:.2f}")
    mt, _, ypt = metrics_from_probs(yt_true, yt_prob, threshold=t_best)
    print(json.dumps(mt, indent=2))
    print(classification_report(yt_true, ypt, target_names=["benign", "malignant"], zero_division=0))

    # Guardado final: sólo pesos para evitar problemas de serialización con AdamW/TFA en TF 2.10
    model.save_weights("cnn7_effnetb0_transfer_final.weights.h5")
    print("\n💾 Pesos guardados: cnn7_effnetb0_transfer_final.weights.h5")


if __name__ == "__main__":
    main()
