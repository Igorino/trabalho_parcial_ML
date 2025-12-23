import os
import json
import joblib
import numpy as np

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# CONFIG
DATA_DIR = "resources/celeba_subset"
OUT_DIR = "results"
MODEL_PATH = "models/model.dat"
CONFIG_PATH = "results/config.txt"
ERROR_PATH = "results/error.txt"
ACC_PATH = "results/acc.txt"

IMG_SIZE = (256, 256) # reduz p/ ficar rápido
TEST_SIZE = 0.3
SEED = 42

# HOG params
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

# Modelo (SVM linear)
MODEL_PARAMS = dict(C=10.0)


def list_images_by_class(root_dir):
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    paths = []
    labels = []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)

    return paths, np.array(labels), class_names


def extract_hog(path):
    img = imread(path)

    # garante 2D
    if img.ndim == 3:
        img = rgb2gray(img)

    img = resize(img, IMG_SIZE, anti_aliasing=True)

    feat = hog(img, **HOG_PARAMS)
    return feat.astype(np.float32)


def save_config(extra=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    config = {
        "img_size": IMG_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "hog_params": HOG_PARAMS,
        "model": "LinearSVC",
        "model_params": MODEL_PARAMS,
    }
    if extra:
        config.update(extra)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, ensure_ascii=False))


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    paths, y, class_names = list_images_by_class(DATA_DIR)
    if len(paths) == 0:
        raise RuntimeError("Não achei imagens.")

    # Extrai features
    X = np.vstack([extract_hog(p) for p in paths])

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treina
    clf = LinearSVC(**MODEL_PARAMS)
    clf.fit(X_train, y_train)

    # Avalia
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Acurácia:", acc)
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # error.txt (salva só um “erro” simbólico)
    # depois vou salva por época
    err = 1.0 - acc
    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        f.write(f"error={err}\n")

    # acc.txt pra eu salvar a acurácia
    with open(ACC_PATH, "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc}\n")

    # model.dat
    joblib.dump({"scaler": scaler, "clf": clf, "classes": class_names}, MODEL_PATH)

    save_config(extra={"num_samples": len(paths), "num_classes": len(class_names)})
    print("\nArquivos gerados:")
    print("-", CONFIG_PATH)
    print("-", ACC_PATH)
    print("-", ERROR_PATH)
    print("-", MODEL_PATH)


if __name__ == "__main__":
    main()
