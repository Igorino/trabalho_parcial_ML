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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from mlp_model import SimpleMLP  # <<< nossa MLP


# CONFIG
DATA_DIR = "resources/celeba_subset"
OUT_DIR = "results_mlp"
MODEL_PATH = "models/model_mlp.dat"
CONFIG_PATH = os.path.join(OUT_DIR, "config.txt")
ERROR_PATH = os.path.join(OUT_DIR, "error.txt")
ACC_PATH = os.path.join(OUT_DIR, "acc.txt")

IMG_SIZE = (256, 256)
TEST_SIZE = 0.3
VAL_SIZE = 0.2
SEED = 42

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

MLP_PARAMS = dict(
    hidden_dim=128,
    lr=1e-2,
    epochs=100,
)


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


def extract_hog(path):
    img = imread(path)

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
        "val_size": VAL_SIZE,
        "seed": SEED,
        "hog_params": HOG_PARAMS,
        "model": "SimpleMLP",
        "mlp_params": MLP_PARAMS,
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

    X = np.vstack([extract_hog(p) for p in paths])
    num_samples, input_dim = X.shape
    num_classes = len(class_names)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    mlp = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=MLP_PARAMS["hidden_dim"],
        output_dim=num_classes,
        lr=MLP_PARAMS["lr"],
        epochs=MLP_PARAMS["epochs"],
        seed=SEED,
    )

    mlp.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Acurácia (teste):", acc)
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        for e, tl, vl, ta, va in zip(
            mlp.history_["epoch"],
            mlp.history_["train_loss"],
            mlp.history_["val_loss"],
            mlp.history_["train_acc"],
            mlp.history_["val_acc"],
        ):
            if vl is None:
                vl = ""
            if va is None:
                va = ""
            f.write(f"{e},{tl},{vl},{ta},{va}\n")

    with open(ACC_PATH, "w", encoding="utf-8") as f:
        f.write(f"accuracy_test={acc}\n")

    joblib.dump(
        {
            "scaler": scaler,
            "mlp_params": MLP_PARAMS,
            "classes": class_names,
            "W1": mlp.W1,
            "b1": mlp.b1,
            "W2": mlp.W2,
            "b2": mlp.b2,
        },
        MODEL_PATH,
    )

    save_config(extra={"num_samples": num_samples, "num_classes": num_classes})

    print("\nArquivos gerados:")
    print("-", CONFIG_PATH)
    print("-", ACC_PATH)
    print("-", ERROR_PATH)
    print("-", MODEL_PATH)


if __name__ == "__main__":
    main()
