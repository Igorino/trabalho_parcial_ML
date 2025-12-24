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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from mlp_model import SimpleMLP  # <<< nossa MLP


# CONFIG
OUT_DIR = "results_mlp"
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
OUT_DIR = "results_mlp"
MODEL_PATH = "models/model_mlp.dat"
CONFIG_PATH = os.path.join(OUT_DIR, "config.txt")
ERROR_PATH = os.path.join(OUT_DIR, "error.txt")
ACC_PATH = os.path.join(OUT_DIR, "acc.txt")

IMG_SIZE = (256, 256)
TEST_SIZE = 0.3 # para teste
VAL_SIZE = 0.2 # vai virar validação
SEED = 42

# HOG params
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

# hiperparâmetros da MLP
MLP_PARAMS = dict(
    hidden_dim=128,
    lr=0.01,
    epochs=100,
    weight_decay=0.0001,
)


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


def extract_hog(path):
    try:
        img = imread(path)

        if img.ndim == 3:
            img = rgb2gray(img)

        img = resize(img, IMG_SIZE, anti_aliasing=True)
        feat = hog(img, **HOG_PARAMS)
        return feat.astype(np.float32)
    except Exception as e:
        logger.error(f"Erro ao processar imagem {path}: {e}")
        raise


def save_config(extra=None):
    logger.info("Salvando configuração...")
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
    logger.info(f"Configuração salva em: {CONFIG_PATH}")


def main():
    start_time = datetime.now()
    logger.info(f"Iniciando treinamento MLP em {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Carrega paths + rótulos
    logger.info("Carregando dataset...")
    paths, y, class_names = list_images_by_class(DATA_DIR)
    if len(paths) == 0:
        raise RuntimeError("Não achei imagens.")

    # 2) Extrai HOG
    logger.info("Extraindo features HOG...")
    features_list = []
    for i, p in enumerate(paths):
        if i % 100 == 0:
            logger.info(f"  Processando imagem {i + 1}/{len(paths)} ({(i + 1) / len(paths) * 100:.1f}%)")
        features_list.append(extract_hog(p))
    
    X = np.vstack(features_list)
    num_samples, input_dim = X.shape
    num_classes = len(class_names)
    logger.info(f"Features extraídas: shape={X.shape}")

    # 3) Split: primeiro treino+val vs teste
    logger.info("Dividindo dataset...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 4) Separa treino vs validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_trainval,
    )
    logger.info(f"  Treino: {X_train.shape[0]} amostras")
    logger.info(f"  Validação: {X_val.shape[0]} amostras")
    logger.info(f"  Teste: {X_test.shape[0]} amostras")

    # 5) Normalização (fit só no treino)
    logger.info("Normalizando features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 6) Cria e treina a MLP
    logger.info("Iniciando treinamento da MLP...")
    mlp = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=MLP_PARAMS["hidden_dim"],
        output_dim=num_classes,
        lr=MLP_PARAMS["lr"],
        epochs=MLP_PARAMS["epochs"],
        seed=SEED,
        weight_decay=MLP_PARAMS["weight_decay"],
    )

    training_start = datetime.now()
    mlp.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    training_time = datetime.now() - training_start
    logger.info(f"Treinamento concluído em {training_time.total_seconds():.2f}s")

    # 7) Avalia no conjunto de teste
    logger.info("Avaliando modelo no teste...")
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Acurácia (teste): {acc:.4f}")
    print("Acurácia (teste):", acc)
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório:")
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print(report)
    
    logger.info("Relatório de classificação:")
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")

    # 8) Salva curva de erro/acurácia por época no error.txt
    logger.info(f"Salvando resultados em {OUT_DIR}...")
    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        f.write(f"Execucao em {start_time.strftime('%d/%m/%Y %H:%M')}\n")
        f.write("epoca;erro_treino;erro_validacao\n")
        for e, tl, vl in zip(
            mlp.history_["epoch"],
            mlp.history_["train_loss"],
            mlp.history_["val_loss"],
        ):
            if vl is None:
                vl = 0.0
            f.write(f"{e};{tl:.6f};{vl:.6f}\n")

    # 9) acc.txt simples com a acurácia final de teste
    with open(ACC_PATH, "w", encoding="utf-8") as f:
        f.write(f"accuracy_test={acc}\n")

    # 10) model_mlp.dat com pesos, + scaler, + classes
    logger.info("Salvando modelo...")
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

    total_time = datetime.now() - start_time
    logger.info(f"Processo concluído em {total_time.total_seconds():.2f}s")

    print("\nArquivos gerados:")
    print("-", CONFIG_PATH)
    print("-", ACC_PATH)
    print("-", ERROR_PATH)
    print("-", MODEL_PATH)
    print(f"- {OUT_DIR}/training.log")


if __name__ == "__main__":
    main()
