import os
import json
import joblib
import numpy as np

from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# CONF
DATA_DIR = "resources/celeba_subset"
OUT_DIR = "results_lbp"
MODEL_PATH = "models/model_lbp.dat"
CONFIG_PATH = os.path.join(OUT_DIR, "config.txt")
ERROR_PATH = os.path.join(OUT_DIR, "error.txt")
ACC_PATH = os.path.join(OUT_DIR, "acc.txt")

IMG_SIZE = (256, 256)
TEST_SIZE = 0.3
SEED = 42

# LBP params
RADIUS = 1
N_POINTS = 8 * RADIUS
LBP_METHOD = "uniform"

# Modelo (SVM linear)
MODEL_PARAMS = dict(C=10.0)  # pode ajustar depois se quiser


def list_images_by_class(root_dir):
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    paths = []
    labels = []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)

    return paths, np.array(labels), class_names


def extract_lbp(path):
    img = imread(path)

    # garante 2D
    if img.ndim == 3:
        img = rgb2gray(img)

    img = resize(img, IMG_SIZE, anti_aliasing=True)
    img = (img * 255).astype("uint8")

    # calcula o LBP da imagem toda
    lbp = local_binary_pattern(img, N_POINTS, RADIUS, method=LBP_METHOD)

    # histograma dos padrões LBP
    # para method="uniform", os valores vão de 0 a N_POINTS + 1
    n_bins = N_POINTS + 2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
    )

    # normaliza para virar distribuição de probabilidade
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist


def save_config(extra=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    config = {
        "img_size": IMG_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "descriptor": "LBP",
        "lbp_params": {
            "radius": RADIUS,
            "n_points": N_POINTS,
            "method": LBP_METHOD,
        },
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
        raise RuntimeError("Não achei imagens em resources/celeba_subset.")

    # Extrai features LBP
    X = np.vstack([extract_lbp(p) for p in paths])

    # Split treino/teste (se quiser, pode tentar stratify=y; se reclamar das classes pequenas, tira)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treina SVM linear
    clf = LinearSVC(**MODEL_PARAMS)
    clf.fit(X_train, y_train)

    # Avalia
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Acurácia (teste) - LBP + SVM:", acc)
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # error.txt – aqui a gente não tem épocas; salva só "erro simbólico"
    err = 1.0 - acc
    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        f.write(f"error={err}\n")

    # acc.txt
    with open(ACC_PATH, "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc}\n")

    # model_lbp.dat
    joblib.dump(
        {
            "scaler": scaler,
            "clf": clf,
            "classes": class_names,
            "lbp_params": {
                "radius": RADIUS,
                "n_points": N_POINTS,
                "method": LBP_METHOD,
            },
        },
        MODEL_PATH,
    )

    save_config(extra={"num_samples": len(paths), "num_classes": len(class_names)})

    print("\nArquivos gerados (LBP):")
    print("-", CONFIG_PATH)
    print("-", ACC_PATH)
    print("-", ERROR_PATH)
    print("-", MODEL_PATH)


if __name__ == "__main__":
    main()
