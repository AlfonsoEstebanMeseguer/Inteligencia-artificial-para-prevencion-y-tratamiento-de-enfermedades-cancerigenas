import os
import json
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# CONFIGURACIÓN Y LOGGING
# ===============================

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===============================
# ENUMS Y DATACLASSES
# ===============================

class AugmentationLevel(Enum):
    """Niveles de aumento de datos"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    ADVANCED = "advanced"
    EXPERT = "expert"

class NormalizationMode(Enum):
    """Modos de normalización"""
    IMAGENET = "imagenet"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    CUSTOM = "custom"
    STANDARD = "standard"  # Normalización a [0,1]

class DataSplit(Enum):
    """Divisiones de datos"""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

@dataclass
class DatasetConfig:
    """Configuración completa del dataset"""
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    buffer_size: int = 1000
    augmentation_level: AugmentationLevel = AugmentationLevel.MEDIUM
    normalization_mode: NormalizationMode = NormalizationMode.IMAGENET
    seed: int = 42
    use_class_weights: bool = True
    cache: bool = True
    shuffle_train: bool = True
    prefetch: bool = True
    
    # Parámetros de augmentación
    rotation_range: float = 15.0  # grados
    zoom_range: Tuple[float, float] = (0.9, 1.1)
    width_shift_range: float = 0.1  # fracción del ancho
    height_shift_range: float = 0.1  # fracción del alto
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    saturation_range: Tuple[float, float] = (0.9, 1.1)
    hue_range: float = 0.1
    
    # Parámetros específicos por nivel
    augmentation_params: Dict[str, float] = None
    
    def __post_init__(self):
        if self.augmentation_params is None:
            self.augmentation_params = self._get_default_augmentation_params()
    
    def _get_default_augmentation_params(self) -> Dict[str, float]:
        """Parámetros por defecto según nivel de augmentación"""
        params = {
            AugmentationLevel.NONE: {
                'rotation': 0.0, 'zoom': 0.0, 'shift': 0.0,
                'brightness': 0.0, 'contrast': 0.0, 'flip_prob': 0.0,
                'hue': 0.0, 'saturation': 0.0, 'cutout_prob': 0.0
            },
            AugmentationLevel.LOW: {
                'rotation': 5.0, 'zoom': 0.05, 'shift': 0.05,
                'brightness': 0.05, 'contrast': 0.05, 'flip_prob': 0.3,
                'hue': 0.02, 'saturation': 0.05, 'cutout_prob': 0.0
            },
            AugmentationLevel.MEDIUM: {
                'rotation': 15.0, 'zoom': 0.1, 'shift': 0.1,
                'brightness': 0.1, 'contrast': 0.1, 'flip_prob': 0.5,
                'hue': 0.05, 'saturation': 0.1, 'cutout_prob': 0.1
            },
            AugmentationLevel.ADVANCED: {
                'rotation': 25.0, 'zoom': 0.2, 'shift': 0.15,
                'brightness': 0.15, 'contrast': 0.15, 'flip_prob': 0.7,
                'hue': 0.1, 'saturation': 0.15, 'cutout_prob': 0.2,
                'mixup_alpha': 0.2, 'cutmix_alpha': 1.0
            },
            AugmentationLevel.EXPERT: {
                # Tratamos EXPERT como experimental: bajar intensidad de color para histología
                'rotation': 30.0, 'zoom': 0.25, 'shift': 0.2,
                'brightness': 0.2, 'contrast': 0.2, 'flip_prob': 0.8,
                'hue': 0.08, 'saturation': 0.12, 'cutout_prob': 0.25,
                'mixup_alpha': 0.4, 'cutmix_alpha': 1.0,
                'gaussian_noise': 0.05, 'speckle_noise': 0.02
            }
        }
        return params.get(self.augmentation_level, params[AugmentationLevel.MEDIUM])

# ===============================
# NORMALIZACIÓN
# ===============================

class ImageNormalizer:
    """Manejador de normalización de imágenes"""
    
    # Estadísticas de ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Estadísticas de ResNet (típicamente las mismas)
    RESNET_MEAN = [0.485, 0.456, 0.406]
    RESNET_STD = [0.229, 0.224, 0.225]
    
    # Estadísticas de EfficientNet
    EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
    EFFICIENTNET_STD = [0.229, 0.224, 0.225]
    
    @classmethod
    def normalize(cls, image: tf.Tensor, mode: NormalizationMode = NormalizationMode.IMAGENET) -> tf.Tensor:
        """Normaliza la imagen según el modo especificado"""
        image = tf.cast(image, tf.float32)
        
        if mode == NormalizationMode.IMAGENET:
            return cls._normalize_imagenet(image)
        elif mode == NormalizationMode.RESNET:
            return cls._normalize_resnet(image)
        elif mode == NormalizationMode.EFFICIENTNET:
            return cls._normalize_efficientnet(image)
        elif mode == NormalizationMode.STANDARD:
            return image / 255.0
        elif mode == NormalizationMode.CUSTOM:
            # Puedes definir tus propias estadísticas aquí
            return image / 255.0
        else:
            raise ValueError(f"Modo de normalización no soportado: {mode}")
    
    @classmethod
    def _normalize_imagenet(cls, image: tf.Tensor) -> tf.Tensor:
        image = image / 255.0
        image = (image - cls.IMAGENET_MEAN) / cls.IMAGENET_STD
        return image
    
    @classmethod
    def _normalize_resnet(cls, image: tf.Tensor) -> tf.Tensor:
        # ResNet50 de Keras espera imágenes en [0,255]
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(image)
    
    @classmethod
    def _normalize_efficientnet(cls, image: tf.Tensor) -> tf.Tensor:
        # EfficientNet de Keras también espera [0,255]
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input(image)

# ===============================
# AUMENTACIÓN DE DATOS PROFESIONAL
# ===============================

class AdvancedAugmenter:
    """Aumentación de datos profesional con múltiples niveles"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.params = config.augmentation_params
        self.rng = tf.random.Generator.from_seed(config.seed)
    
    def apply_augmentations(self, image: tf.Tensor, label: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica aumentaciones según el nivel configurado"""
        if not training or self.config.augmentation_level == AugmentationLevel.NONE:
            return image, label
        
        # Trabajamos en rango [0,1] para operaciones de color/ruido
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Aplicar aumentaciones básicas según nivel
        image = self._apply_basic_augmentations(image)
        
        # Aplicar aumentaciones avanzadas si corresponde
        if self.config.augmentation_level in [AugmentationLevel.ADVANCED, AugmentationLevel.EXPERT]:
            image, label = self._apply_advanced_augmentations(image, label)
        
        return image, label
    
    def _apply_basic_augmentations(self, image: tf.Tensor) -> tf.Tensor:
        """Aplica aumentaciones básicas"""
        
        # Rotación aleatoria
        if self.params.get('rotation', 0) > 0:
            angle = self.rng.uniform([], -self.params['rotation'], self.params['rotation'])
            angle = angle * (np.pi / 180.0)  # Convertir a radianes
            image = tf.keras.layers.RandomRotation(factor=self.params['rotation'] / 360.0)(image)
        
        # Zoom aleatorio
        if self.params.get('zoom', 0) > 0:
            zoom_factor = self.rng.uniform([], 1 - self.params['zoom'], 1 + self.params['zoom'])
            image = tf.keras.layers.RandomZoom(
                height_factor=self.params['zoom'],
                width_factor=self.params['zoom']
            )(image)
        
        # Desplazamiento aleatorio
        if self.params.get('shift', 0) > 0:
            image = tf.keras.layers.RandomTranslation(
                height_factor=self.params['shift'],
                width_factor=self.params['shift']
            )(image)
        
        # Volteos aleatorios
        if self.params.get('flip_prob', 0) > 0:
            # Horizontal
            if self.rng.uniform([]) < self.params['flip_prob']:
                image = tf.image.flip_left_right(image)
            # Vertical (solo para histología si tiene sentido)
            if self.rng.uniform([]) < self.params['flip_prob'] / 2:
                image = tf.image.flip_up_down(image)
        
        # Ajuste de color
        if self.params.get('brightness', 0) > 0:
            image = tf.image.random_brightness(image, max_delta=self.params['brightness'])
        
        if self.params.get('contrast', 0) > 0:
            image = tf.image.random_contrast(
                image, 
                lower=1 - self.params['contrast'],
                upper=1 + self.params['contrast']
            )
        
        if self.params.get('hue', 0) > 0:
            image = tf.image.random_hue(image, max_delta=self.params['hue'])
        
        if self.params.get('saturation', 0) > 0:
            image = tf.image.random_saturation(
                image,
                lower=1 - self.params['saturation'],
                upper=1 + self.params['saturation']
            )
        
        # Recorte aleatorio (solo si la imagen es suficientemente grande)
        if self.config.img_size[0] > 100 and self.config.img_size[1] > 100:
            crop_size = int(min(self.config.img_size) * 0.9)
            image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
            image = tf.image.resize(image, self.config.img_size)
        
        return image
    
    def _apply_advanced_augmentations(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica aumentaciones avanzadas (MixUp, CutMix, CutOut, etc.)"""
        
        # CutOut / Random Erasing
        if self.params.get('cutout_prob', 0) > 0 and self.rng.uniform([]) < self.params['cutout_prob']:
            image = self._apply_cutout(image)
        
        # Añadir ruido gaussiano (solo en nivel EXPERT)
        if self.config.augmentation_level == AugmentationLevel.EXPERT and self.params.get('gaussian_noise', 0) > 0:
            noise = self.rng.normal(tf.shape(image), mean=0.0, stddev=self.params['gaussian_noise'])
            image = image + noise
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Añadir ruido speckle (solo en nivel EXPERT)
        if self.config.augmentation_level == AugmentationLevel.EXPERT and self.params.get('speckle_noise', 0) > 0:
            noise = self.rng.normal(tf.shape(image), mean=0.0, stddev=self.params['speckle_noise'])
            image = image + image * noise
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def _apply_cutout(self, image: tf.Tensor) -> tf.Tensor:
        """Aplica CutOut (Random Erasing)"""
        h, w = self.config.img_size
        mask_size_h = int(h * 0.2)  # 20% de la altura
        mask_size_w = int(w * 0.2)  # 20% del ancho
        
        # Posición aleatoria para el recorte
        y = self.rng.uniform([], 0, h - mask_size_h, dtype=tf.int32)
        x = self.rng.uniform([], 0, w - mask_size_w, dtype=tf.int32)
        
        # Crear máscara
        mask = tf.ones([mask_size_h, mask_size_w, 3])
        
        # Aplicar máscara
        paddings = [[y, h - (y + mask_size_h)], [x, w - (x + mask_size_w)], [0, 0]]
        mask = tf.pad(mask, paddings, mode='CONSTANT', constant_values=1)
        
        # Invertir máscara y aplicar
        mask = 1 - mask
        image = image + mask
        
        return tf.clip_by_value(image, 0.0, 1.0)

# ===============================
# MANEJO DE DATOS
# ===============================

class BreakHisDataLoader:
    """Cargador profesional de datos BreakHis"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.augmenter = AdvancedAugmenter(config)
        self.class_weights = None
        self.label_encoder = {'benign': 0, 'malignant': 1}
        
    def load_split(self, split_dir: str, split_type: DataSplit) -> Tuple[List[str], List[int]]:
        """Carga un split desde archivo JSON"""
        json_path = os.path.join(split_dir, f"{split_type.value}.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No se encontró el archivo: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        images = data.get("images", [])
        labels = data.get("labels", [])
        
        # Verificar consistencia
        if len(images) != len(labels):
            raise ValueError(f"Inconsistencia en {json_path}: {len(images)} imágenes vs {len(labels)} etiquetas")
        
        logger.info(f"✅ {split_type.value.upper()}: {len(images)} imágenes cargadas")
        return images, labels
    
    def compute_class_weights(self, train_labels: List[int]) -> Dict[int, float]:
        """Calcula pesos de clase para manejar desbalanceo"""
        unique_classes = np.unique(train_labels)
        
        if len(unique_classes) != 2:
            logger.warning(f"Se esperaban 2 clases pero se encontraron {len(unique_classes)}")
        
        # Calcular pesos usando sklearn
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        # Convertir a diccionario
        self.class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        logger.info(f"⚖️  Pesos de clase calculados: {self.class_weights}")
        logger.info(f"   Ratio Maligno/Benigno: {self.class_weights.get(1, 1.0)/self.class_weights.get(0, 1.0):.2f}:1")
        
        return self.class_weights
    
    def decode_image(self, img_path: tf.Tensor) -> tf.Tensor:
        """Decodifica una imagen desde la ruta"""
        def _decode(path: tf.Tensor) -> tf.Tensor:
            """Decodifica en CPU; si falla devuelve imagen en negro."""
            try:
                img_bytes = tf.io.read_file(path)
                img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
                return img
            except Exception as e:
                logger.warning(f"No se pudo decodificar: {path.numpy().decode('utf-8')} ({e})")
                return tf.zeros((*self.config.img_size, 3), dtype=tf.uint8)

        img = tf.py_function(_decode, [img_path], Tout=tf.uint8)
        img.set_shape([None, None, 3])
        return img
    
    def preprocess_image(self, img: tf.Tensor, label: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Pipeline completo de preprocesamiento"""
        # Convertir a float32 y redimensionar
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, self.config.img_size)

        # Aumentación (solo en training) antes de normalizar
        if training:
            img01 = tf.clip_by_value(img / 255.0, 0.0, 1.0)
            img01, label = self.augmenter.apply_augmentations(img01, label, training)
            img = tf.clip_by_value(img01 * 255.0, 0.0, 255.0)

        # Normalizar al final
        img = ImageNormalizer.normalize(img, self.config.normalization_mode)

        return img, label
    
    def create_dataset(self, image_paths: List[str], labels: List[int], training: bool = False) -> tf.data.Dataset:
        """Crea un tf.data.Dataset"""
        
        # Convertir a tensores
        paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        # Combinar
        dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
        
        # Decodificar imágenes
        dataset = dataset.map(
            lambda path, label: (self.decode_image(path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Preprocesamiento
        dataset = dataset.map(
            lambda img, label: self.preprocess_image(img, label, training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Optimizaciones
        if training and self.config.shuffle_train:
            dataset = dataset.shuffle(
                buffer_size=min(len(image_paths), self.config.buffer_size),
                reshuffle_each_iteration=True,
                seed=self.config.seed
            )
        
        if self.config.cache:
            dataset = dataset.cache()
        
        dataset = dataset.batch(self.config.batch_size)
        
        if training:
            # Repetir indefinidamente: usar steps_per_epoch en model.fit para evitar loops infinitos
            dataset = dataset.repeat()
        
        if self.config.prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_datasets(self, split_dir: str) -> Dict[str, tf.data.Dataset]:
        """Crea todos los datasets (train, val, test)"""
        
        datasets = {}
        
        # Cargar todos los splits
        splits_data = {}
        for split in [DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST]:
            try:
                images, labels = self.load_split(split_dir, split)
                splits_data[split] = (images, labels)
            except FileNotFoundError as e:
                logger.warning(f"No se pudo cargar {split}: {e}")
                splits_data[split] = ([], [])
        
        # Calcular pesos de clase usando train
        train_images, train_labels = splits_data[DataSplit.TRAIN]
        if train_labels and self.config.use_class_weights:
            self.compute_class_weights(train_labels)
        
        # Crear datasets
        for split, (images, labels) in splits_data.items():
            if images:  # Solo crear si hay datos
                training = (split == DataSplit.TRAIN)
                datasets[split.value] = self.create_dataset(images, labels, training)
                
                logger.info(f"📦 Dataset {split.value} creado: {len(images)} imágenes")
        
        return datasets
    
    def visualize_augmentations(self, image_path: str, num_samples: int = 5):
        """Visualiza aumentaciones para debugging"""
        if not os.path.exists(image_path):
            logger.error(f"Imagen no encontrada: {image_path}")
            return
        
        # Cargar imagen original
        original_img = self.decode_image(image_path)
        original_img_norm = ImageNormalizer.normalize(original_img, self.config.normalization_mode)
        
        # Crear figura
        fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))
        
        # Mostrar original
        axes[0].imshow(tf.cast(original_img_norm * 255, tf.uint8).numpy())
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Mostrar aumentaciones
        label = tf.constant(0)  # Etiqueta dummy
        for i in range(num_samples):
            img_aug, _ = self.augmenter.apply_augmentations(original_img_norm, label, training=True)
            axes[i+1].imshow(tf.cast(img_aug * 255, tf.uint8).numpy())
            axes[i+1].set_title(f"Aug {i+1}")
            axes[i+1].axis('off')
        
        plt.suptitle(f"Nivel de Augmentación: {self.config.augmentation_level.value}", fontsize=14)
        plt.tight_layout()
        
        # Guardar figura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"augmentation_visualization_{timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Visualización guardada en: {save_path}")
        
        return fig

# ===============================
# FUNCIÓN PRINCIPAL
# ===============================

def create_breakhis_pipeline(
    split_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augmentation_level: str = "medium",
    normalization_mode: str = "imagenet",
    use_class_weights: bool = True,
    seed: int = 42
) -> Tuple[Dict[str, tf.data.Dataset], Optional[Dict[int, float]]]:
    """
    Crea un pipeline completo de datos para BreakHis
    
    Args:
        split_dir: Directorio con los archivos JSON de splits
        img_size: Tamaño de las imágenes
        batch_size: Tamaño del batch
        augmentation_level: Nivel de augmentación ('none', 'low', 'medium', 'advanced', 'expert')
        normalization_mode: Modo de normalización
        use_class_weights: Si calcular pesos de clase
        seed: Semilla para reproducibilidad
    
    Returns:
        Tuple con: dict de datasets y pesos de clase
    """
    
    # Validar parámetros
    try:
        aug_level = AugmentationLevel(augmentation_level.lower())
    except ValueError:
        logger.warning(f"Nivel de augmentación '{augmentation_level}' no válido. Usando 'medium'")
        aug_level = AugmentationLevel.MEDIUM
    
    try:
        norm_mode = NormalizationMode(normalization_mode.lower())
    except ValueError:
        logger.warning(f"Modo de normalización '{normalization_mode}' no válido. Usando 'imagenet'")
        norm_mode = NormalizationMode.IMAGENET
    
    # Crear configuración
    config = DatasetConfig(
        img_size=img_size,
        batch_size=batch_size,
        augmentation_level=aug_level,
        normalization_mode=norm_mode,
        seed=seed,
        use_class_weights=use_class_weights
    )

    if aug_level == AugmentationLevel.EXPERT:
        logger.warning("⚠️  Augmentación EXPERT es experimental para histología. Considera usar MEDIUM/ADVANCED en TFG.")
    
    logger.info("=" * 60)
    logger.info("🚀 CONFIGURACIÓN DEL PIPELINE DE DATOS")
    logger.info("=" * 60)
    logger.info(f"   • Tamaño imagen: {img_size}")
    logger.info(f"   • Batch size: {batch_size}")
    logger.info(f"   • Augmentación: {augmentation_level.upper()}")
    logger.info(f"   • Normalización: {normalization_mode.upper()}")
    logger.info(f"   • Pesos de clase: {'Sí' if use_class_weights else 'No'}")
    logger.info(f"   • Semilla: {seed}")
    
    # Crear cargador de datos
    loader = BreakHisDataLoader(config)
    
    # Crear datasets
    datasets = loader.create_datasets(split_dir)
    
    # Resumen final
    logger.info("=" * 60)
    logger.info("📊 RESUMEN FINAL DE DATASETS")
    logger.info("=" * 60)
    for split_name, dataset in datasets.items():
        # Contar elementos (aproximado)
        try:
            count = sum(1 for _ in dataset)
            logger.info(f"   • {split_name.upper()}: {count} batches")
        except:
            logger.info(f"   • {split_name.upper()}: Dataset creado")
    
    return datasets, loader.class_weights

# ===============================
# EJEMPLO DE USO
# ===============================

if __name__ == "__main__":
    
    # Configuración
    SPLIT_DIR = "splits"  # Directorio con train.json, val.json, test.json
    
    # Crear pipeline con diferentes configuraciones
    try:
        # Configuración experta
        datasets, class_weights = create_breakhis_pipeline(
            split_dir=SPLIT_DIR,
            img_size=(256, 256),  # Un poco más grande para mejor detalle
            batch_size=16,  # Batch más pequeño para GPUs con menos memoria
            augmentation_level="expert",  # Nivel máximo de augmentación
            normalization_mode="imagenet",
            use_class_weights=True,
            seed=42
        )
        
        # Acceder a los datasets
        train_ds = datasets.get("train")
        val_ds = datasets.get("val")
        test_ds = datasets.get("test")
        
        # Sanity check
        if train_ds:
            logger.info("\n🧪 SANITY CHECK - Primer batch de entrenamiento:")
            for images, labels in train_ds.take(1):
                logger.info(f"   • Forma imágenes: {images.shape}")
                logger.info(f"   • Rango imágenes: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
                logger.info(f"   • Labels (primeros 5): {labels.numpy()[:5]}")
                logger.info(f"   • Distribución labels: {np.unique(labels.numpy(), return_counts=True)}")
        
        if class_weights:
            logger.info(f"\n⚖️  Pesos de clase para training:")
            for class_id, weight in class_weights.items():
                class_name = "Benigno" if class_id == 0 else "Maligno"
                logger.info(f"   • {class_name} (clase {class_id}): peso = {weight:.3f}")
        
        # Visualizar augmentaciones (opcional)
        # Buscar una imagen de ejemplo
        import glob
        example_images = glob.glob(os.path.join(SPLIT_DIR, "*.jpg")) + glob.glob(os.path.join(SPLIT_DIR, "*.png"))
        if example_images:
            # Crear un loader para visualización
            viz_config = DatasetConfig(
                img_size=(224, 224),
                augmentation_level=AugmentationLevel.EXPERT
            )
            viz_loader = BreakHisDataLoader(viz_config)
            viz_loader.visualize_augmentations(example_images[0], num_samples=5)
        
    except Exception as e:
        logger.error(f"❌ Error creando pipeline: {e}")
        import traceback
        traceback.print_exc()
