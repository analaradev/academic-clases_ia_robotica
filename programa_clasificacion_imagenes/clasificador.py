import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

# =========================
# CONFIGURACION
# =========================
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15
MODEL_PATH = "modelo_clasificacion.keras"
CLASSES_FILE = "clases.txt"


def entrenar():
    print("=" * 50)
    print("ENTRENAMIENTO DE CLASIFICADOR DE OBJETOS")
    print("=" * 50)
    print("")

    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: No se encontro la carpeta '{DATASET_DIR}'.")
        print("Organiza tus imagenes en subcarpetas por clase dentro de 'dataset/'.")
        return

    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\nClases detectadas: {class_names}")
    print(f"Total de clases: {num_classes}")
    print("")

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Creando modelo con MobileNetV2 (transfer learning)...")

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    print("")
    print("Iniciando entrenamiento...")
    print("")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print("")
    print("Evaluando modelo...")
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Perdida: {loss:.4f}")
    print(f"Precision: {accuracy * 100:.2f}%")

    model.save(MODEL_PATH)

    with open(CLASSES_FILE, "w") as f:
        for c in class_names:
            f.write(c + "\n")

    print("")
    print(f"Modelo guardado como: {MODEL_PATH}")
    print(f"Clases guardadas en: {CLASSES_FILE}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validacion")
    plt.title("Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validacion")
    plt.title("Perdida")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("grafica_entrenamiento.png")
    print("Grafica guardada como: grafica_entrenamiento.png")
    print("")
    print("Entrenamiento completado.")


def cargar_modelo_y_clases():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encontro '{MODEL_PATH}'. Entrena el modelo primero con: python clasificador.py entrenar")
        sys.exit(1)

    if not os.path.exists(CLASSES_FILE):
        print(f"ERROR: No se encontro '{CLASSES_FILE}'.")
        sys.exit(1)

    with open(CLASSES_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    model = keras.models.load_model(MODEL_PATH)
    return model, class_names


def predecir_webcam():
    model, class_names = cargar_modelo_y_clases()
    num_classes = len(class_names)

    np.random.seed(42)
    colores = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (num_classes, 3))]

    print("Clases cargadas:", class_names)
    print("Modelo listo.")
    print("")
    print("Instrucciones:")
    print("- Muestra un objeto a la camara.")
    print("- Presiona 'q' para salir.")
    print("")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        img_resized = cv2.resize(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype("float32") / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        prediccion = model.predict(img_batch, verbose=0)[0]
        clase_idx = np.argmax(prediccion)
        confianza = prediccion[clase_idx]
        nombre_clase = class_names[clase_idx]

        texto = f"{nombre_clase.upper()}: {confianza * 100:.1f}%"
        color = colores[clase_idx]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, texto, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        top3_indices = np.argsort(prediccion)[-3:][::-1]
        y_offset = 100
        for i, idx in enumerate(top3_indices):
            texto_top = f"{i+1}. {class_names[idx]}: {prediccion[idx] * 100:.1f}%"
            cv2.putText(frame, texto_top, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colores[idx], 1)
            y_offset += 30

        cv2.imshow("Clasificador de Imagenes - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")


def predecir_imagen(img_path):
    model, class_names = cargar_modelo_y_clases()

    if not os.path.exists(img_path):
        print(f"Error: no se encontro '{img_path}'")
        sys.exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: no se pudo leer '{img_path}'")
        sys.exit(1)

    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype("float32") / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    prediccion = model.predict(img_batch, verbose=0)[0]
    clase_idx = np.argmax(prediccion)
    confianza = prediccion[clase_idx]

    print("")
    print("=" * 40)
    print("RESULTADO DE LA PREDICCION")
    print("=" * 40)
    print(f"Clase: {class_names[clase_idx].upper()}")
    print(f"Confianza: {confianza * 100:.2f}%")
    print("=" * 40)
    print("")
    print("Top 3 clases mas probables:")
    top3_indices = np.argsort(prediccion)[-3:][::-1]
    for i, idx in enumerate(top3_indices):
        print(f"  {i+1}. {class_names[idx]}: {prediccion[idx] * 100:.2f}%")

    resultado_texto = f"{class_names[clase_idx].upper()}: {confianza * 100:.1f}%"
    cv2.putText(img, resultado_texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# MENU PRINCIPAL
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python clasificador.py <comando> [argumentos]")
        print("")
        print("Comandos disponibles:")
        print("  entrenar                          Entrena el modelo con las imagenes de dataset/")
        print("  webcam                            Clasifica objetos en tiempo real con la webcam")
        print("  imagen <ruta_de_la_imagen>        Clasifica una imagen guardada")
        print("")
        print("Ejemplos:")
        print("  python clasificador.py entrenar")
        print("  python clasificador.py webcam")
        print("  python clasificador.py imagen dataset/botellas/botella 1.jpg")
        sys.exit(1)

    comando = sys.argv[1].lower()

    if comando == "entrenar":
        entrenar()
    elif comando == "webcam":
        predecir_webcam()
    elif comando == "imagen":
        if len(sys.argv) < 3:
            print("ERROR: Debes especificar la ruta de la imagen.")
            print("Ejemplo: python clasificador.py imagen foto.jpg")
            sys.exit(1)
        predecir_imagen(sys.argv[2])
    else:
        print(f"Comando desconocido: '{comando}'")
        print("Usa: entrenar, webcam o imagen")
        sys.exit(1)
