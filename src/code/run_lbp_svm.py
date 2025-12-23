import os
import json
import joblib
import numpy as np
import logging
from datetime import datetime

from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# CONF
OUT_DIR = "results_lbp"
os.makedirs(OUT_DIR, exist_ok=True)

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Listando imagens no diretório: {root_dir}")
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    logger.info(f"Encontradas {len(class_names)} classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    paths = []
    labels = []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        count = 0
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)
                count += 1
        logger.info(f"  Classe {idx} ({cls}): {count} imagens")

    logger.info(f"Total: {len(paths)} imagens, {len(class_names)} classes")
    return paths, np.array(labels), class_names


def extract_lbp(path):
    try:
        img = imread(path)

        # garante 2D
        if img.ndim == 3:
            img = rgb2gray(img)

        img = resize(img, IMG_SIZE, anti_aliasing=True)
        img = (img * 255).astype("uint8")

        # calcula o LBP da imagem toda
        lbp = local_binary_pattern(img, N_POINTS, RADIUS, method=LBP_METHOD)

        # histograma dos padrões LBP
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
    except Exception as e:
        logger.error(f"Erro ao processar imagem {path}: {e}")
        raise


def save_config(extra=None):
    logger.info("Salvando configuração...")
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
    logger.info(f"Configuração salva em: {CONFIG_PATH}")


def main():
    start_time = datetime.now()
    logger.info(f"Iniciando treinamento LBP+SVM em {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Carregando dataset...")
    paths, y, class_names = list_images_by_class(DATA_DIR)
    if len(paths) == 0:
        raise RuntimeError("Não achei imagens em resources/celeba_subset.")

    # Extrai features LBP
    logger.info("Extraindo features LBP...")
    features_list = []
    for i, p in enumerate(paths):
        if i % 100 == 0:
            logger.info(f"  Processando imagem {i + 1}/{len(paths)} ({(i + 1) / len(paths) * 100:.1f}%)")
        features_list.append(extract_lbp(p))
    
    X = np.vstack(features_list)
    logger.info(f"Features extraídas: shape={X.shape}")

    # Split treino/teste (se quiser, pode tentar stratify=y; se reclamar das classes pequenas, tira)
    logger.info("Dividindo dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    logger.info(f"  Treino: {X_train.shape[0]} amostras")
    logger.info(f"  Teste: {X_test.shape[0]} amostras")

    # Normalização
    logger.info("Normalizando features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treina SVM linear
    logger.info("Treinando modelo LinearSVC...")
    clf = LinearSVC(**MODEL_PARAMS)
    
    training_start = datetime.now()
    clf.fit(X_train, y_train)
    training_time = datetime.now() - training_start
    logger.info(f"Treinamento concluído em {training_time.total_seconds():.2f}s")

    # Avalia
    logger.info("Avaliando modelo...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Acurácia final: {acc:.4f}")
    print("Acurácia (teste) - LBP + SVM:", acc)
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório:")
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print(report)

    logger.info("Relatório de classificação:")
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")

    # error.txt – aqui a gente não tem épocas; salva só "erro simbólico"
    logger.info(f"Salvando resultados em {OUT_DIR}...")
    err = 1.0 - acc
    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        f.write(f"error={err}\n")

    # acc.txt
    with open(ACC_PATH, "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc}\n")

    # model_lbp.dat
    logger.info("Salvando modelo...")
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

    total_time = datetime.now() - start_time
    logger.info(f"Processo concluído em {total_time.total_seconds():.2f}s")

    print("\nArquivos gerados (LBP):")
    print("-", CONFIG_PATH)
    print("-", ACC_PATH)
    print("-", ERROR_PATH)
    print("-", MODEL_PATH)
    print(f"- {OUT_DIR}/training.log")


if __name__ == "__main__":
    main()
