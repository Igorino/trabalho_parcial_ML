import os
import json
import joblib
import numpy as np
import logging
from datetime import datetime

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CONFIG
DATA_DIR = "resources/celeba_subset"
OUT_DIR = "results"
MODEL_PATH = "models/model.dat"
CONFIG_PATH = "results/config.txt"
ERROR_PATH = "results/error.txt"
ACC_PATH = "results/acc.txt"

IMG_SIZE = (256, 256)  # reduz p/ ficar rápido (alterado pra 256 pra melhorar um pouco acurácia)
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
MODEL_PARAMS = dict(C=10.0)  # Alterado pra 10 pra melhorar acurácia


def list_images_by_class(root_dir):
    logger.info(f" Listando imagens no diretório: {root_dir}")

    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    logger.info(f"  Encontradas {len(class_names)} classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

    paths = []
    labels = []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        class_images = []

        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(idx)
                class_images.append(fn)

        logger.info(f"   Classe {idx} ({cls}): {len(class_images)} imagens")

    logger.info(f" Total: {len(paths)} imagens, {len(class_names)} classes")
    return paths, np.array(labels), class_names


def extract_hog(path):
    try:
        img = imread(path)

        # garante 2D
        if img.ndim == 3:
            img = rgb2gray(img)

        img = resize(img, IMG_SIZE, anti_aliasing=True)

        feat = hog(img, **HOG_PARAMS)
        return feat.astype(np.float32)
    except Exception as e:
        logger.error(f" Erro ao processar imagem {path}: {e}")
        raise


def save_config(extra=None):
    logger.info("💾 Salvando configuração...")
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

    logger.info(f" Configuração salva em: {CONFIG_PATH}")


def main():
    start_time = datetime.now()
    logger.info(f" Iniciando treinamento em {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        logger.info(" Carregando dataset...")
        paths, y, class_names = list_images_by_class(DATA_DIR)
        if len(paths) == 0:
            raise RuntimeError("Não achei imagens.")

        logger.info(" Extraindo features HOG...")
        features_list = []
        for i, path in enumerate(paths):
            if i % 100 == 0:  # Log a cada 100 imagens
                logger.info(f"   Processando imagem {i + 1}/{len(paths)} ({(i + 1) / len(paths) * 100:.1f}%)")

            feat = extract_hog(path)
            features_list.append(feat)

        X = np.vstack(features_list)
        logger.info(f" Features extraídas: shape={X.shape}")

        logger.info(" Dividindo dataset...")
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )

        logger.info(f"   Treino: {X_train.shape[0]} amostras")
        logger.info(f"   Teste: {X_test.shape[0]} amostras")

        # Verifica distribuição das classes
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        logger.info(
            f"   Distribuição treino: min={counts_train.min()}, max={counts_train.max()}, média={counts_train.mean():.1f}")
        logger.info(
            f"   Distribuição teste: min={counts_test.min()}, max={counts_test.max()}, média={counts_test.mean():.1f}")

        logger.info(" Normalizando features...")
        # Normalização
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logger.info(f"   Features normalizadas: média={X_train.mean():.3f}, std={X_train.std():.3f}")

        logger.info("Treinando modelo LinearSVC...")
        logger.info(f"   Parâmetros: {MODEL_PARAMS}")
        logger.info(f"   Dados de treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        logger.info(f"   Número de classes: {len(np.unique(y_train))}")

        # Treina
        clf = LinearSVC(verbose=1, **MODEL_PARAMS)  # verbose=1 para ver progresso interno
        training_start = datetime.now()

        logger.info("Iniciando treinamento... (isso pode demorar alguns minutos)")

        clf.fit(X_train, y_train)

        training_time = datetime.now() - training_start

        logger.info(f" Treinamento concluído em {training_time.total_seconds():.2f}s")

        logger.info(" Avaliando modelo...")
        # Avalia
        y_pred_train = clf.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        logger.info(f" Acurácia treino: {acc_train:.4f}")
        logger.info(f" Acurácia teste: {acc:.4f} ({acc * 100:.2f}%)")
        print("Acurácia:", acc)
        print("Matriz de confusão:")
        print(confusion_matrix(y_test, y_pred))

        print("\nRelatório:")
        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        print(report)

        # Log do relatório também
        logger.info(" Relatório de classificação:")
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"   {line}")

        logger.info(" Salvando resultados...")

        # error.txt
        err_train = 1.0 - acc_train
        err_test = 1.0 - acc
        with open(ERROR_PATH, "w", encoding="utf-8") as f:
            f.write(f"Execucao em {start_time.strftime('%d/%m/%Y %H:%M')}\n")
            f.write("epoca;erro_treino;erro_validacao\n")
            f.write(f"1;{err_train:.6f};{err_test:.6f}\n")

        # acc.txt pra eu salvar a acurácia
        with open(ACC_PATH, "w", encoding="utf-8") as f:
            f.write(f"accuracy={acc}\n")

        # model.dat
        logger.info(" Salvando modelo...")
        joblib.dump({"scaler": scaler, "clf": clf, "classes": class_names}, MODEL_PATH)

        save_config(extra={
            "num_samples": len(paths),
            "num_classes": len(class_names),
            "training_time_seconds": training_time.total_seconds(),
            "final_accuracy": float(acc)
        })

        total_time = datetime.now() - start_time
        logger.info(f" Processo concluído em {total_time.total_seconds():.2f}s")

        print("\nArquivos gerados:")
        print("-", CONFIG_PATH)
        print("-", ACC_PATH)
        print("-", ERROR_PATH)
        print("-", MODEL_PATH)
        print("- results/training.log")

    except Exception as e:
        logger.error(f" Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()