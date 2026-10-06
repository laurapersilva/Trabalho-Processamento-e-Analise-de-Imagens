"""
Trabalho Prático - Classificação de pCR em DCE-MRI
Grupo: 
Integrantes: Laura Persilva, Arthur Signorini, Andriel Mark e Otávio Monteiro.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from scipy.ndimage import rotate
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO (TODO: ajustar conforme a estrutura real do dataset/CSV)
# ----------------------------------------------------------------------------
DATA_DIR = "BreastDCEDL_ISPY1_min_crop"
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")   # TODO: nome real do csv
COL_ID, COL_LABEL, COL_SPLIT = "patient_id", "pCR", "split"  # TODO: conferir colunas
PHASES = ["pre", "early", "late"]    # TODO: mapear para os nomes de arquivo reais
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def phase_path(pid, phase):
    """Caminho do .nii de uma fase. TODO: adaptar ao padrão de nomes do dataset."""
    return os.path.join(DATA_DIR, "images", f"{pid}_{phase}.nii.gz")


def mask_path(pid):
    """Caminho da segmentação. TODO: adaptar."""
    return os.path.join(DATA_DIR, "masks", f"{pid}_mask.nii.gz")


# ----------------------------------------------------------------------------
# LEITURA E NORMALIZAÇÃO
# ----------------------------------------------------------------------------
def load_nii(path):
    """Lê um volume NIfTI e retorna array (H, W, Z). Confira a orientação no MRIcro!"""
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def normalize(vol):
    """Normalização min-max por volume. TODO: avaliar z-score / percentis."""
    lo, hi = np.percentile(vol, (1, 99))
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)


# ----------------------------------------------------------------------------
# AUMENTO DE DADOS (item b): 5 ângulos x (original, espelhada) = 10 variações
# ----------------------------------------------------------------------------
ANGLES = [-20, -10, 0, 10, 20]


def augment(img):
    """img: (C, H, W). Retorna lista com 10 variações."""
    out = []
    for flip in (False, True):
        base = img[:, :, ::-1] if flip else img
        for ang in ANGLES:
            if ang == 0:
                out.append(base.copy())
            else:
                out.append(rotate(base, ang, axes=(1, 2), reshape=False, order=1))
    return out


def class_weights(labels):
    """Peso da classe positiva para BCEWithLogits. TODO: verificar o desbalanceamento
    e decidir/justificar a estratégia (pos_weight, focal loss, aumento proporcional)."""
    labels = np.asarray(labels)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


# ----------------------------------------------------------------------------
# DATASET (item c: estratégia de seleção de corte)
# ----------------------------------------------------------------------------
class DCEDataset(Dataset):
    """
    phases: lista de fases usadas (1 fase -> 1 canal; 3 fases -> 3 canais).
    slice_mode: 'center' | 'max_tumor'  (TODO: implementar 'all' com média/máximo)
    """

    def __init__(self, df, phases, slice_mode="center", augment_data=False):
        self.df, self.phases = df.reset_index(drop=True), phases
        self.slice_mode, self.augment_data = slice_mode, augment_data
        # TODO: se augment_data, expandir o índice em 10 variações por amostra
        # (ou aplicar on-the-fly, mas justificar no relatório).

    def __len__(self):
        return len(self.df)

    def pick_slice(self, vols, mask):
        if self.slice_mode == "center":
            return vols[0].shape[2] // 2
        if self.slice_mode == "max_tumor":
            return int(np.argmax(mask.sum(axis=(0, 1))))
        raise ValueError(self.slice_mode)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        pid = row[COL_ID]
        vols = [normalize(load_nii(phase_path(pid, p))) for p in self.phases]
        mask = load_nii(mask_path(pid))
        z = self.pick_slice(vols, mask)
        x = np.stack([v[:, :, z] for v in vols])            # (C, H, W)
        if len(self.phases) == 1:                            # ResNet/CNN: replicar se preciso
            pass  # TODO: decidir entre 1 canal ou replicar para 3 (ResNet50)
        y = float(row[COL_LABEL])
        return torch.from_numpy(x.copy()).float(), torch.tensor(y)


# ----------------------------------------------------------------------------
# MODELOS (item d)
# ----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """CNN própria. Requisito: >= 5 camadas COM parâmetros (conv/linear;
    entrada e pooling não contam). TODO: ajustar e justificar hiperparâmetros."""

    def __init__(self, in_ch=3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_ch, 16), block(16, 32), block(32, 64),
                                      block(64, 128), block(128, 128))   # 5 convs
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5),
                                        nn.Linear(128, 64), nn.ReLU(),
                                        nn.Linear(64, 1))                # + 2 lineares

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))  # logit


def build_resnet50(in_ch=3):
    """ResNet50 com pesos ImageNet; congela o backbone e retreina a parte FC."""
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters():
        p.requires_grad = False
    if in_ch != 3:  # TODO: adaptar conv1 (ou replicar o canal) se usar 1 fase
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 1)  # saída binária
    return m


def build_custom_model(in_ch=3):
    """TODO: terceira solução à escolha (ex.: 3 ramos independentes + fusão,
    EfficientNet/DenseNet, atenção...)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# TREINO / AVALIAÇÃO
# ----------------------------------------------------------------------------
def train(model, train_loader, val_loader, pos_weight, epochs=30, lr=1e-4,
          save_path="model.pt"):
    """TODO: early stopping, scheduler, registro da loss por época (gráfico de
    convergência), tempo de treino e salvamento do melhor modelo."""
    model.to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
        val_loss = evaluate_loss(model, val_loader, crit)
        print(f"época {ep+1}: val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), save_path)


@torch.no_grad()
def evaluate_loss(model, loader, crit):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        tot += crit(model(x).squeeze(1), y).item() * len(y)
        n += len(y)
    return tot / max(n, 1)


def metrics(y_true, y_prob, thr=0.5):
    """TODO: sensibilidade, especificidade, precisão, acurácia, F1."""
    raise NotImplementedError


def aggregate(probs, mode="mean"):
    """Agrega previsões por corte (item c). 'mean' ou 'max'."""
    return float(np.mean(probs) if mode == "mean" else np.max(probs))


# ----------------------------------------------------------------------------
# GRAD-CAM (item e)
# ----------------------------------------------------------------------------
def grad_cam(model, x, target_layer):
    """TODO: registrar hooks na camada alvo (ex.: model.layer4[-1] na ResNet50),
    obter ativações e gradientes do logit, ponderar canais, ReLU, upsample e
    sobrepor na imagem."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# INTERFACE GRÁFICA (item a) - tkinter + Pillow (sem matplotlib)
# ----------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DCE-MRI pCR - Visualizador")
        self.vol = self.seg = None
        self.z = tk.IntVar(value=0)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=6)
        ttk.Button(bar, text="Abrir imagem (.nii)", command=self.open_img).pack(side="left")
        ttk.Button(bar, text="Abrir segmentação", command=self.open_seg).pack(side="left", padx=4)
        # TODO: botões para treinar, carregar modelo, classificar e Grad-CAM

        view = ttk.Frame(self)
        view.pack()
        self.lbl_img, self.lbl_seg = ttk.Label(view), ttk.Label(view)
        self.lbl_img.grid(row=0, column=0, padx=4)
        self.lbl_seg.grid(row=0, column=1, padx=4)

        self.slider = ttk.Scale(self, from_=0, to=0, orient="horizontal",
                                command=lambda v: self.show(int(float(v))))
        self.slider.pack(fill="x", padx=6)
        self.info = ttk.Label(self, text="")
        self.info.pack()

    def open_img(self):
        p = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if p:
            self.vol = normalize(load_nii(p))
            self.slider.configure(to=self.vol.shape[2] - 1)
            self.show(0)

    def open_seg(self):
        p = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if p:
            self.seg = load_nii(p)
            self.show(int(float(self.slider.get())))

    @staticmethod
    def to_photo(arr2d):
        # TODO: conferir orientação (transpor/espelhar) comparando com o MRIcro
        a = (np.clip(arr2d, 0, 1) * 255).astype(np.uint8).T
        return ImageTk.PhotoImage(Image.fromarray(a).resize((384, 384)))

    def show(self, z):
        if self.vol is None:
            return
        z = min(z, self.vol.shape[2] - 1)
        self._p1 = self.to_photo(self.vol[:, :, z])
        self.lbl_img.configure(image=self._p1)
        if self.seg is not None and z < self.seg.shape[2]:
            self._p2 = self.to_photo((self.seg[:, :, z] > 0).astype(np.float32))
            self.lbl_seg.configure(image=self._p2)
        self.info.configure(text=f"corte z = {z + 1}/{self.vol.shape[2]}")


if __name__ == "__main__":
    App().mainloop()
