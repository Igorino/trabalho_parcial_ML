import os
import random
import shutil
from collections import defaultdict

# CONFIG
IMG_DIR = "/home/igor/Downloads/img_align_celeba"
IDENTITY_FILE = "resources/annotations/identity_CelebA.txt"
OUT_DIR = "resources/celeba_subset"
K_IDS = 50
M_PER_ID = 10
SEED = 42

USE_SYMLINK = False  # True = rápido e não duplica espaço | False = copia

def safe_link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if USE_SYMLINK:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)

def main():
    random.seed(SEED)

    # id -> [img1, img2, ...]
    id_to_imgs = defaultdict(list)

    with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fn, id_str = line.split()
            id_to_imgs[int(id_str)].append(fn)

    # filtra ids com pelo menos M fotos
    candidates = [i for i, imgs in id_to_imgs.items() if len(imgs) >= M_PER_ID]
    if len(candidates) < K_IDS:
        raise RuntimeError(f"Poucos ids com >= {M_PER_ID} fotos. Achou só {len(candidates)}.")

    chosen_ids = random.sample(candidates, K_IDS)

    # limpa saída (opcional)
    os.makedirs(OUT_DIR, exist_ok=True)

    total = 0
    for person_id in chosen_ids:
        imgs = id_to_imgs[person_id]
        chosen_imgs = random.sample(imgs, M_PER_ID)

        person_dir = os.path.join(OUT_DIR, f"person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)

        for fn in chosen_imgs:
            src = os.path.join(IMG_DIR, fn)
            dst = os.path.join(person_dir, fn)
            safe_link(src, dst)
            total += 1

    print("Subset criado!")
    print("Pasta:", OUT_DIR)
    print("IDs:", len(chosen_ids))
    print("Imagens totais:", total)
    print("Symlink:", USE_SYMLINK)

if __name__ == "__main__":
    main()
