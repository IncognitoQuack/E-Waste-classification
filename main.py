# main.py
# End-to-end E-Waste classifier: teacher → distill → prune → quantize → serve

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses, optimizers, backend as K
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow_model_optimization.sparsity import keras as sparsity
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr     
import json
import csv

# ── Config ─────────────────────────────────────────────────────
IMG_SIZE       = 128
BATCH_SIZE     = 32
EPOCHS_TEACHER = 15
EPOCHS_STUDENT = 10
LR_INITIAL     = 1e-4
TEMPERATURE    = 5.0
DISTILL_ALPHA  = 0.3
PRUNE_SPARSITY = 0.5
DATA_DIR       = "data/E-Waste classification dataset/modified-dataset"
OUTPUT_DIR     = "outputs"
SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Data & Augmentations ───────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def mixup(images, labels, alpha=0.2):
    """Applies MixUp on a batch."""
    batch_size = tf.shape(images)[0]
    lam = np.random.beta(alpha, alpha)
    idx = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, idx)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, idx)
    return mixed_images, mixed_labels

def prepare_dataset(split, augment=False):
    """Loads tf.data.Dataset for train/val/test, with optional MixUp+standard aug."""
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        label_mode="categorical",
        image_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=(split=="train"),
        seed=SEED
    )
    def _rescale(x,y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    ds = ds.map(_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x,y: tf.cond(
                        tf.random.uniform([],0,1)<0.5,
                        lambda: mixup(x,y),
                        lambda: (x,y)
                     ),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# ── Model Components ────────────────────────────────────────────
def se_block(x, reduction=16):
    """Squeeze-and-Excitation block."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters//reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1,1,filters))(se)
    return layers.multiply([x, se])

def build_teacher(input_shape, num_classes):
    base = EfficientNetV2B0(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = se_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def build_student(input_shape, num_classes):
    base = EfficientNetV2B0(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

# ── Losses & Callbacks ───────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1-y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=1)
    return loss

def cosine_annealing(epoch):
    lr_min = 1e-6
    return lr_min + 0.5*(LR_INITIAL-lr_min)*(1 + tf.cos(np.pi * epoch / EPOCHS_TEACHER))

# ── Distiller ───────────────────────────────────────────────────
class Distiller(models.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, student_loss_fn, distill_loss_fn, metrics):
        super().compile()
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.distill_metric = metrics

    def train_step(self, data):
        x, y = data
        teacher_pred = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            s_loss = self.student_loss_fn(y, student_pred)
            t_soft = tf.nn.softmax(teacher_pred / TEMPERATURE, axis=1)
            s_soft = tf.nn.softmax(student_pred / TEMPERATURE, axis=1)
            d_loss = self.distill_loss_fn(t_soft, s_soft) * (TEMPERATURE**2)
            loss = DISTILL_ALPHA * s_loss + (1 - DISTILL_ALPHA) * d_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.distill_metric.update_state(y, student_pred)
        return {"loss": loss, "student_loss": s_loss, "distill_loss": d_loss,
                "accuracy": self.distill_metric.result()}

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        s_loss = self.student_loss_fn(y, student_pred)
        self.distill_metric.update_state(y, student_pred)
        return {"loss": s_loss, "accuracy": self.distill_metric.result()}

# ── Training / Distillation / Pruning / Quantization ────────────
def train_teacher(ds_train, ds_val, inp_shape, n_classes):
    model = build_teacher(inp_shape, n_classes)
    model.compile(
        optimizer=optimizers.Adam(LR_INITIAL),
        loss=focal_loss(),
        metrics=["accuracy"]
    )
    cb = [
      callbacks.ModelCheckpoint(f"{OUTPUT_DIR}/teacher_best.h5",
                                monitor="val_accuracy", save_best_only=True),
      callbacks.LearningRateScheduler(cosine_annealing),
      callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    model.fit(
        prepare_dataset("train", augment=True),
        validation_data=prepare_dataset("val"),
        epochs=EPOCHS_TEACHER,
        callbacks=cb
    )
    return model

def train_student(ds_train, ds_val, teacher, inp_shape, n_classes):
    student = build_student(inp_shape, n_classes)
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=optimizers.Adam(LR_INITIAL),
        student_loss_fn=losses.CategoricalCrossentropy(),
        distill_loss_fn=losses.KLDivergence(),
        metrics=tf.keras.metrics.CategoricalAccuracy()
    )
    cb = [
      callbacks.ModelCheckpoint(f"{OUTPUT_DIR}/student_best.h5",
                                monitor="accuracy", save_best_only=True),
      callbacks.EarlyStopping(monitor="accuracy", patience=3, restore_best_weights=True)
    ]
    distiller.fit(
        prepare_dataset("train", augment=True),
        validation_data=prepare_dataset("val"),
        epochs=EPOCHS_STUDENT,
        callbacks=cb
    )
    return student

def apply_pruning(model, ds_train):
    # schedule from 0→PRUNE_SPARSITY over total steps
    total_steps = np.ceil((len(ds_train)*EPOCHS_STUDENT)/BATCH_SIZE).astype(int)
    pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(
           initial_sparsity=0.0,
           final_sparsity=PRUNE_SPARSITY,
           begin_step=0,
           end_step=total_steps)
    }
    pruned = sparsity.prune_low_magnitude(model, **pruning_params)
    pruned.compile(
      optimizer=optimizers.Adam(LR_INITIAL/10),
      loss=losses.CategoricalCrossentropy(),
      metrics=["accuracy"]
    )
    pruned.fit(
      prepare_dataset("train", augment=True),
      epochs=2,
      callbacks=[sparsity.UpdatePruningStep()]
    )
    return sparsity.strip_pruning(pruned)

def export_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite)
    print(f"TFLite model saved to {path}")

# ── Evaluation & Gradio ─────────────────────────────────────────
def evaluate(model, ds_test, class_names):
    y_true, y_pred = [], []
    for x,y in ds_test:
        p = model.predict(x)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    # save metrics
    with open(f"{OUTPUT_DIR}/metrics.csv","w") as f:
        writer=csv.writer(f)
        writer.writerow(["class","precision","recall","f1-score","support"])
        for cls in class_names:
            m=report[cls]
            writer.writerow([cls,m["precision"],m["recall"],m["f1-score"],m["support"]])
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:\n", cm)

def serve(model, class_names):
    def predict(img):
        img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE)) / 255.0
        pred = model.predict(img[None,...])[0]
        return {n: float(pred[i]) for i,n in enumerate(class_names)}
    gr.Interface(fn=predict,
                 inputs=gr.Image(shape=(IMG_SIZE,IMG_SIZE)),
                 outputs=gr.Label(num_top_classes=3),
                 title="E-Waste Classifier").launch()

# ── Orchestration ───────────────────────────────────────────────
if __name__ == "__main__":
    # load once to infer shape & classes
    ds_train = prepare_dataset("train")
    ds_val   = prepare_dataset("val")
    ds_test  = prepare_dataset("test")
    class_names = ds_train.element_spec[1].shape[-1]  # or ds_train.class_names if available

    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    n_classes  = ds_train.element_spec[1].shape[-1]

    print("→ Training teacher")
    teacher = train_teacher(ds_train, ds_val, input_shape, n_classes)

    print("→ Distilling student")
    student = train_student(ds_train, ds_val, teacher, input_shape, n_classes)

    print("→ Pruning student")
    pruned = apply_pruning(student, ds_train)

    print("→ Evaluating final model")
    evaluate(pruned, ds_test, [str(i) for i in range(n_classes)])

    print("→ Exporting TFLite")
    export_tflite(pruned, os.path.join(OUTPUT_DIR, "model.tflite"))

    print("→ Launching Gradio demo")
    serve(pruned, [str(i) for i in range(n_classes)])
