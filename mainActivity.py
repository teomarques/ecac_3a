# -*- coding: utf-8 -*-
"""
EA/ECAC 2025 - Trabalho Prático 1
mainActivity.py

Bernardo Direito - 2023215454
Teodoro Marques  - 2023211717
"""
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

# --- Constantes Globais ---
COL_DEVICE_ID = 0
COL_ACC_X = 1
COL_ACC_Y = 2
COL_ACC_Z = 3
COL_GYRO_X = 4
COL_GYRO_Y = 5
COL_GYRO_Z = 6
COL_MAG_X = 7
COL_MAG_Y = 8
COL_MAG_Z = 9
COL_TIMESTAMP = 10
COL_ACTIVITY = 11
COL_PARTICIPANT = 12

MODULO_INDICES = [
    (COL_ACC_X, COL_ACC_Y, COL_ACC_Z),
    (COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z),
    (COL_MAG_X, COL_MAG_Y, COL_MAG_Z)
]

VAR_LABELS = [
    "Módulo Acelerômetro",
    "Módulo Giroscópio",
    "Módulo Magnetômetro"
]
SENSOR_LABELS = {
    1: "Pulso Esquerdo",
    2: "Pulso Direito",
    3: "Peito",
    4: "Perna Sup. Direita",
    5: "Perna Inf. Esquerda"
}

ATIVIDADES_META2 = [1, 2, 3, 4, 5, 6, 7]
ATIVIDADE_META2_MAX = 7


# --- Funções baseadas no embeddings_extractor.py ---

def load_model():
    print("A carregar modelo HARNet5 (necessita de internet na 1ª execução)...")
    repo = 'OxWearables/ssl-wearables'
    model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
    model.eval()
    feature_encoder = model.feature_extractor
    feature_encoder.to("cpu")
    feature_encoder.eval()
    return feature_encoder

def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    fs_target = 30.0
    win_size = 5
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)
    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])
    return acc_resampled, fs_target

# -----------------------------------

# ///// META 1 /////

def carregar_dados_participante(num_participante, base_dir="."):
    part_dir = os.path.join(base_dir, f"part{num_participante}")
    dados = []
    for i in range(1, 6):
        caminho_ficheiro = os.path.join(part_dir, f"part{num_participante}dev{i}.csv")
        try:
            with open(caminho_ficheiro, newline='') as csvfile:
                leitor = csv.reader(csvfile)
                for linha in leitor:
                    try:
                        dados.append([float(x) for x in linha])
                    except ValueError:
                        continue
        except FileNotFoundError:
            print(f"[Aviso] Ficheiro não encontrado: {caminho_ficheiro}")
        except Exception as e:
            print(f"[Erro] Problema ao ler {caminho_ficheiro}: {e}")

    if not dados:
        return np.array([])
    return np.array(dados)

def carregar_dados_todos_participantes(base_dir="."):
    dados_todos = []
    for part_num in range(15):
        print(f"A carregar participante {part_num}...")
        dados_participante = carregar_dados_participante(part_num, base_dir)
        if dados_participante.size > 0:
            dados_todos.append(dados_participante)

    if not dados_todos:
        print("[Erro Fatal] Nenhum dado carregado.")
        return np.array([])
    return np.concatenate(dados_todos, axis=0)

def carregar_dados_todos_com_id(base_dir="."):
    dados_todos = []
    for part_num in range(15):
        print(f"A carregar participante {part_num}...")
        dados_p = carregar_dados_participante(part_num, base_dir)
        if dados_p.size > 0:
            col_id = np.full((dados_p.shape[0], 1), part_num)
            dados_p = np.hstack([dados_p, col_id])
            dados_todos.append(dados_p)
    
    if not dados_todos:
        print("[Erro Fatal] Nenhum dado carregado.")
        return np.array([])
    return np.concatenate(dados_todos, axis=0)

def calcular_modulo_vetor(dados, col_x, col_y, col_z):
    """
    Calcula o módulo (norma Euclidiana) de um vetor tridimensional.
    """
    vetor = dados[:, [col_x, col_y, col_z]]
    modulo = np.linalg.norm(vetor, axis=1)
    return modulo

def get_dados_transformados(dados):
    print("A calcular módulos (dataset transformado)...")
    ids_sensores = dados[:, COL_DEVICE_ID]
    ids_atividades = dados[:, COL_ACTIVITY]
    modulos = np.zeros((dados.shape[0], 3))

    for i, (cols) in enumerate(MODULO_INDICES):
        modulos[:, i] = calcular_modulo_vetor(dados, cols[0], cols[1], cols[2])

    dados_transformados = np.stack([
        ids_sensores,
        ids_atividades,
        modulos[:, 0],
        modulos[:, 1],
        modulos[:, 2]
    ], axis=1)
    return dados_transformados

def plotar_boxplots_3_1(dados_transformados):
    print("A gerar gráficos para Tarefa 3.1...")
    fig, axes = plt.subplots(3, 5, figsize=(24, 15), sharex=True)
    fig.suptitle("Tarefa 3.1: Módulos por Atividade e Dispositivo", fontsize=20, fontweight='bold')

    for var_idx in range(3):
        for sensor_id in range(1, 6):
            ax = axes[var_idx, sensor_id-1]
            sensor_mask = dados_transformados[:, 0] == sensor_id
            dados_sensor = dados_transformados[sensor_mask]
            dados_box = []
            for a in range(1, 17):
                activity_mask = dados_sensor[:, 1] == a
                dados_ativ = dados_sensor[activity_mask, var_idx + 2]
                dados_box.append(dados_ativ)
            ax.boxplot(dados_box, tick_labels=[str(a) for a in range(1, 17)])
            if var_idx == 0: ax.set_title(SENSOR_LABELS[sensor_id], fontsize=14, fontweight='bold')
            if sensor_id == 1: ax.set_ylabel(VAR_LABELS[var_idx], fontsize=14, fontweight='bold')
            ax.set_xlabel("Atividade")
            ax.tick_params(axis='x', rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("graphs/meta1_tarefa_3_1_boxplots.png")
    print("Gráfico guardado.")

def analisar_densidade_iqr_3_2(dados_transformados):
    print("\n--- Início Tarefa 3.2: Densidade de Outliers (IQR - Pulso Direito) ---")
    sensor_id_foco = 2
    sensor_mask = dados_transformados[:, 0] == sensor_id_foco
    dados_sensor = dados_transformados[sensor_mask]
    print(f"Analisando Sensor: {SENSOR_LABELS[sensor_id_foco]}\n")
    print("| Variável              | Atividade | n_total (nr) | n_outliers (no) | Densidade (d) |")

    for var_idx in range(3):
        var_label = VAR_LABELS[var_idx]
        for a in range(1, 17):
            activity_mask = dados_sensor[:, 1] == a
            amostras = dados_sensor[activity_mask, var_idx + 2]
            n_r = len(amostras)
            if n_r == 0: continue
            q1 = np.percentile(amostras, 25)
            q3 = np.percentile(amostras, 75)
            iqr = q3 - q1
            limite_inf = q1 - (1.5 * iqr)
            limite_sup = q3 + (1.5 * iqr)
            outliers_mask = (amostras < limite_inf) | (amostras > limite_sup)
            n_o = np.sum(outliers_mask)
            densidade = (n_o / n_r) * 100
            print(f"| {var_label:21} | {a:9} | {n_r:12} | {n_o:15} | {densidade:13.2f}% |")

def identificar_outliers_zscore_3_3(amostras, k):
    if amostras.size == 0: return np.array([], dtype=bool)
    media = np.mean(amostras)
    std = np.std(amostras)
    if std == 0: return np.zeros(amostras.shape, dtype=bool)
    z_scores = (amostras - media) / std
    return np.abs(z_scores) > k

def plotar_outliers_zscore_3_4(dados_transformados):
    print("A gerar gráficos para Tarefa 3.4...")
    k_valores = [3, 3.5, 4]
    for k in k_valores:
        print(f"  A gerar para k={k}...")
        fig, axes = plt.subplots(3, 5, figsize=(24, 15), sharex=True)
        fig.suptitle(f"Tarefa 3.4: Outliers Z-Score (k={k})", fontsize=20, fontweight='bold')
        for var_idx in range(3):
            for sensor_id in range(1, 6):
                ax = axes[var_idx, sensor_id-1]
                for a in range(1, 17):
                    sensor_mask = dados_transformados[:, 0] == sensor_id
                    activity_mask = dados_transformados[:, 1] == a
                    mask = sensor_mask & activity_mask
                    amostras = dados_transformados[mask, var_idx + 2]
                    if amostras.size == 0: continue
                    outlier_mask = identificar_outliers_zscore_3_3(amostras, k)
                    inliers = amostras[~outlier_mask]
                    outliers = amostras[outlier_mask]
                    x_base = a
                    x_inliers = np.random.normal(x_base, 0.1, size=inliers.size)
                    x_outliers = np.random.normal(x_base, 0.1, size=outliers.size)
                    ax.scatter(x_inliers, inliers, color='blue', alpha=0.3, s=5)
                    ax.scatter(x_outliers, outliers, color='red', alpha=1.0, s=10)
                if var_idx == 0: ax.set_title(SENSOR_LABELS[sensor_id], fontsize=14, fontweight='bold')
                if sensor_id == 1: ax.set_ylabel(VAR_LABELS[var_idx], fontsize=14, fontweight='bold')
                ax.set_xticks(range(1, 17))
                ax.set_xticklabels([str(a) for a in range(1, 17)], rotation=90)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f"graphs/meta1_tarefa_3_4_zscore_k{k}.png")
        print(f"Gráfico guardado.")

def comparar_densidades_3_5(dados_transformados):
    print("\n--- Início Tarefa 3.5: Comparação Densidades ---")
    sensor_id_foco = 2
    k_valores = [3, 3.5, 4]
    sensor_mask = dados_transformados[:, 0] == sensor_id_foco
    dados_sensor = dados_transformados[sensor_mask]
    print(f"Analisando Sensor: {SENSOR_LABELS[sensor_id_foco]}\n")
    for var_idx in range(3):
        var_label = VAR_LABELS[var_idx]
        for a in range(1, 17):
            activity_mask = dados_sensor[:, 1] == a
            amostras = dados_sensor[activity_mask, var_idx + 2]
            n_r = len(amostras)
            if n_r == 0: continue
            q1 = np.percentile(amostras, 25); q3 = np.percentile(amostras, 75); iqr = q3 - q1
            n_o_iqr = np.sum((amostras < q1 - 1.5*iqr) | (amostras > q3 + 1.5*iqr))
            d_iqr = (n_o_iqr / n_r) * 100
            densidades_z = []
            for k in k_valores:
                n_o_z = np.sum(identificar_outliers_zscore_3_3(amostras, k))
                densidades_z.append((n_o_z / n_r) * 100)
            print(f"| {var_label:21} | {a:4} | {n_r:7} | {d_iqr:13.2f} | {densidades_z[0]:16.2f} | {densidades_z[1]:18.2f} | {densidades_z[2]:16.2f} |")

def kmeans_3_6(X, n_clusters, max_iter=100, tol=1e-4, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], n_clusters, replace=False)
    centroides = X[indices]
    for _ in range(max_iter):
        distancias = np.sqrt(((X[:, np.newaxis] - centroides) ** 2).sum(axis=2))
        labels = np.argmin(distancias, axis=1)
        novos_centroides = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.all(np.linalg.norm(novos_centroides - centroides, axis=1) < tol): break
        centroides = novos_centroides
    return centroides, labels

def analisar_outliers_kmeans_3_7(dados, n_clusters_lista):
    print(f"\n--- Início Tarefa 3.7: Outliers K-Means ---")
    var_idx_foco = 0; sensor_id_foco = 2; atividade_foco = 4
    cols_xyz = MODULO_INDICES[var_idx_foco]
    mask = (dados[:, COL_DEVICE_ID] == sensor_id_foco) & (dados[:, COL_ACTIVITY] == atividade_foco)
    X = dados[mask][:, [cols_xyz[0], cols_xyz[1], cols_xyz[2]]]
    if X.shape[0] < max(n_clusters_lista): return
    for n_clusters in n_clusters_lista:
        try:
            centroides, labels = kmeans_3_6(X, n_clusters, random_state=42)
        except ValueError: continue
        distancias = np.linalg.norm(X - centroides[labels], axis=1)
        limite = np.percentile(distancias, 99)
        outlier_mask = distancias > limite
        inlier_mask = ~outlier_mask
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2], c=labels[inlier_mask], cmap='tab10', alpha=0.6, s=10)
        ax.scatter(X[outlier_mask, 0], X[outlier_mask, 1], X[outlier_mask, 2], c='red', s=50)
        ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c='black', s=200, marker='X')
        ax.set_title(f"Tarefa 3.7: K-Means (k={n_clusters})")
        plt.savefig(f"graphs/meta1_tarefa_3_7_kmeans_k{n_clusters}.png")
        print(f"Gráfico guardado.")

def analisar_outliers_dbscan_3_7_1(dados):
    print(f"\n--- Início Tarefa 3.7.1 (Bónus): Outliers DBSCAN ---")
    var_idx_foco = 0; sensor_id_foco = 2; atividade_foco = 4
    cols_xyz = MODULO_INDICES[var_idx_foco]
    mask = (dados[:, COL_DEVICE_ID] == sensor_id_foco) & (dados[:, COL_ACTIVITY] == atividade_foco)
    X = dados[mask][:, [cols_xyz[0], cols_xyz[1], cols_xyz[2]]]
    if X.shape[0] < 50: return
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5).fit(X)
        dists, _ = nn.kneighbors(X)
        eps = np.percentile(dists[:, -1], 95)
        if eps <= 0: eps = 0.5
    except Exception: eps = 0.5
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    outlier_mask = db.labels_ == -1
    inlier_mask = ~outlier_mask
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2], c=db.labels_[inlier_mask], cmap='tab10', alpha=0.6, s=10)
    ax.scatter(X[outlier_mask, 0], X[outlier_mask, 1], X[outlier_mask, 2], c='red', s=50)
    plt.savefig("graphs/meta1_tarefa_3_7_1_dbscan.png")
    print("Gráfico guardado.")

def testar_significancia_medias_4_1(dados_transformados):
    print("\n--- Início Tarefa 4.1: Testes de Significância ---")
    for sensor_id in range(1, 6):
        sensor_mask = dados_transformados[:, 0] == sensor_id
        dados_sensor = dados_transformados[sensor_mask]
        print(f"\nSensor: {SENSOR_LABELS[sensor_id]}")
        for var_idx in range(3):
            var_label = VAR_LABELS[var_idx]
            series = []
            for a in range(1, 17):
                vals = dados_sensor[dados_sensor[:, 1] == a, var_idx + 2]
                if vals.size >= 8: series.append(vals)
            if len(series) >= 2:
                try:
                    f_stat, p_val = stats.f_oneway(*series)
                    print(f"  {var_label:20} -> ANOVA p={p_val:.3e}")
                except:
                    pass

def _windows_indices_4_2(n_samples, win_size, hop_size):
    for start in range(0, n_samples - win_size + 1, hop_size):
        yield start, start + win_size

def segmentar_janelas_puras_4_2(dados_raw, fs=51.2, janela_s=5.0, overlap=0.5):
    win_size = int(janela_s * fs)
    hop_size = int(win_size * (1.0 - overlap))
    segmentos = []
    for sensor_id in range(1, 6):
        mask_dev = dados_raw[:, COL_DEVICE_ID] == sensor_id
        dados_dev = dados_raw[mask_dev]
        if dados_dev.size == 0: continue
        idx_sort = np.argsort(dados_dev[:, COL_TIMESTAMP])
        dados_dev = dados_dev[idx_sort]
        atividades = dados_dev[:, COL_ACTIVITY].astype(int)
        change_idx = np.where(np.diff(atividades) != 0)[0] + 1
        boundaries = np.concatenate(([0], change_idx, [len(atividades)]))
        for b in range(len(boundaries) - 1):
            i0, i1 = boundaries[b], boundaries[b + 1]
            bloco = dados_dev[i0:i1]
            if bloco.shape[0] < win_size: continue
            atividade = int(bloco[0, COL_ACTIVITY])
            X9 = np.stack([
                bloco[:, COL_ACC_X], bloco[:, COL_ACC_Y], bloco[:, COL_ACC_Z],
                bloco[:, COL_GYRO_X], bloco[:, COL_GYRO_Y], bloco[:, COL_GYRO_Z],
                bloco[:, COL_MAG_X], bloco[:, COL_MAG_Y], bloco[:, COL_MAG_Z],
            ], axis=1)
            for s, e in _windows_indices_4_2(bloco.shape[0], win_size, hop_size):
                seg = X9[s:e]
                if seg.shape[0] == win_size:
                    segmentos.append({'device': sensor_id, 'activity': atividade, 'X': seg})
    return segmentos

def segmentar_janelas_com_sujeito(dados_raw, fs=51.2, janela_s=5.0, overlap=0.5):
    win_size = int(janela_s * fs)
    hop_size = int(win_size * (1.0 - overlap))
    segmentos = []
    sujeitos_unicos = np.unique(dados_raw[:, COL_PARTICIPANT])
    for subj_id in sujeitos_unicos:
        mask_subj = dados_raw[:, COL_PARTICIPANT] == subj_id
        dados_subj = dados_raw[mask_subj]
        for sensor_id in range(1, 6):
            mask_dev = dados_subj[:, COL_DEVICE_ID] == sensor_id
            dados_dev = dados_subj[mask_dev]
            if dados_dev.size == 0: continue
            idx_sort = np.argsort(dados_dev[:, COL_TIMESTAMP])
            dados_dev = dados_dev[idx_sort]
            atividades = dados_dev[:, COL_ACTIVITY].astype(int)
            change_idx = np.where(np.diff(atividades) != 0)[0] + 1
            boundaries = np.concatenate(([0], change_idx, [len(atividades)]))
            for b in range(len(boundaries) - 1):
                i0, i1 = boundaries[b], boundaries[b + 1]
                bloco = dados_dev[i0:i1]
                if bloco.shape[0] < win_size: continue
                atividade = int(bloco[0, COL_ACTIVITY])
                X9 = np.stack([
                    bloco[:, COL_ACC_X], bloco[:, COL_ACC_Y], bloco[:, COL_ACC_Z],
                    bloco[:, COL_GYRO_X], bloco[:, COL_GYRO_Y], bloco[:, COL_GYRO_Z],
                    bloco[:, COL_MAG_X], bloco[:, COL_MAG_Y], bloco[:, COL_MAG_Z],
                ], axis=1)
                for s, e in _windows_indices_4_2(bloco.shape[0], win_size, hop_size):
                    seg = X9[s:e]
                    if seg.shape[0] == win_size:
                        segmentos.append({'device': sensor_id, 'activity': atividade, 'subject': int(subj_id), 'X': seg})
    return segmentos

def _feature_vector_series_temporal_freq_4_2(series, fs):
    feats = []
    mean = np.mean(series)
    std = np.std(series)
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    rms = np.sqrt(np.mean(series ** 2))
    wl = np.sum(np.abs(np.diff(series)))
    zc = np.sum(series[:-1] * series[1:] < 0) / (len(series) - 1)
    feats += [mean, std, median, mad, rms, wl, zc]
    f, Pxx = signal.periodogram(series, fs=fs, scaling='spectrum', window='hann')
    Pxx = np.maximum(Pxx, 1e-12)
    energy = np.sum(Pxx)
    Pnorm = Pxx / np.sum(Pxx)
    spec_entropy = -np.sum(Pnorm * np.log(Pnorm))
    dom_idx = np.argmax(Pxx)
    dom_freq = f[dom_idx]
    dom_amp = Pxx[dom_idx]
    spec_centroid = np.sum(f * Pxx) / np.sum(Pxx)
    feats += [energy, spec_entropy, dom_freq, dom_amp, spec_centroid]
    return feats

def extrair_features_janela_4_2(seg, fs=51.2):
    X = seg['X']
    acc_mod = np.linalg.norm(X[:, 0:3], axis=1)
    gyr_mod = np.linalg.norm(X[:, 3:6], axis=1)
    mag_mod = np.linalg.norm(X[:, 6:9], axis=1)
    series_list = [
        ("acc_x", X[:, 0]), ("acc_y", X[:, 1]), ("acc_z", X[:, 2]),
        ("gyr_x", X[:, 3]), ("gyr_y", X[:, 4]), ("gyr_z", X[:, 5]),
        ("mag_x", X[:, 6]), ("mag_y", X[:, 7]), ("mag_z", X[:, 8]),
        ("acc_mod", acc_mod), ("gyr_mod", gyr_mod), ("mag_mod", mag_mod),
    ]
    feat_vec = []
    feat_names = []
    base = f"dev{seg['device']}"
    for name, s in series_list:
        feats = _feature_vector_series_temporal_freq_4_2(s, fs)
        feat_vec.extend(feats)
        feat_names.extend([
            f"{base}_{name}_mean", f"{base}_{name}_std", f"{base}_{name}_median", f"{base}_{name}_mad", f"{base}_{name}_rms", f"{base}_{name}_wl", f"{base}_{name}_zcr",
            f"{base}_{name}_spec_energy", f"{base}_{name}_spec_entropy", f"{base}_{name}_spec_dom_freq", f"{base}_{name}_spec_dom_amp", f"{base}_{name}_spec_centroid",
        ])
    return np.array(feat_vec, dtype=float), feat_names

def construir_feature_set_4_2(dados_raw, fs=51.2):
    segmentos = segmentar_janelas_puras_4_2(dados_raw, fs=fs, janela_s=5.0, overlap=0.5)
    X_list, y_list = [], []
    feat_names_ref = None
    for seg in segmentos:
        fv, fn = extrair_features_janela_4_2(seg, fs)
        if feat_names_ref is None: feat_names_ref = fn
        X_list.append(fv)
        y_list.append(seg['activity'])
    if not X_list: return np.zeros((0, 0)), np.array([]), []
    return np.vstack(X_list), np.array(y_list, dtype=int), feat_names_ref

def pca_pipeline_4_3_4_4(X, variancia_target=0.75):
    if X.shape[0] == 0: return None, None, None, None
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(Xz)
    cvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cvar, variancia_target) + 1)
    pca_n = PCA(n_components=n_comp)
    Xp = pca_n.fit_transform(Xz)
    return scaler, pca_n, Xp, cvar

def guardar_artifacts_4_x(X, y, feature_names, cvar=None, pca_explained=None):
    try:
        np.savetxt("tarefa4_features_X.csv", X, delimiter=",")
        np.savetxt("tarefa4_labels_y.csv", y, fmt="%d", delimiter=",")
        with open("tarefa4_feature_names.csv", "w") as f:
            f.write("feature\n")
            for n in feature_names: f.write(f"{n}\n")
        if cvar is not None:
            plt.figure(figsize=(6,4))
            plt.plot(np.arange(1, len(cvar)+1), cvar, marker='o')
            plt.axhline(0.75, color='red', linestyle='--')
            plt.tight_layout()
            plt.savefig("graphs/tarefa4_pca_variancia.png")
    except Exception as e:
        print(f"[Aviso] Falha ao guardar artefactos: {e}")

def guardar_rankings_4_6(scores, feature_names, fname):
    try:
        order = np.argsort(np.abs(scores))[::-1]
        with open(fname, "w") as f:
            f.write("rank,feature,score,abs_score\n")
            for r, j in enumerate(order, 1):
                f.write(f"{r},{feature_names[j]},{scores[j]},{abs(scores[j])}\n")
    except Exception: pass

def selecionar_features_topk_4_6(X, feature_names, scores, topk=10):
    order = np.argsort(np.abs(scores))[::-1][:topk]
    X_sel = X[:, order]
    names_sel = [feature_names[j] for j in order]
    return X_sel, names_sel, order

def fisher_score_multi_4_5(X, y):
    n_classes = np.unique(y)
    mu = X.mean(axis=0)
    scores = np.zeros(X.shape[1], dtype=float)
    for c in n_classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        if nc == 0: continue
        mu_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0) + 1e-12
        scores += nc * (mu_c - mu) ** 2 / var_c
    return scores

def relieff_4_5(X, y, n_neighbors=10, n_samples=2000, random_state=0):
    rng = np.random.default_rng(random_state)
    m = X.shape[0]
    if m == 0: return np.zeros(X.shape[1])
    idx_samples = rng.choice(m, size=min(n_samples, m), replace=False)
    W = np.zeros(X.shape[1], dtype=float)
    classes = np.unique(y)
    from sklearn.neighbors import NearestNeighbors
    nbrs_all = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    for i in idx_samples:
        x_i = X[i : i + 1]
        yi = y[i]
        distances, indices = nbrs_all.kneighbors(x_i)
        indices = indices.flatten()[1:]
        same = indices[y[indices] == yi][:n_neighbors]
        for c in classes:
            if c == yi: continue
            idx_c = indices[y[indices] == c][:n_neighbors]
            if idx_c.size == 0: continue
            diff = np.abs(X[i] - X[idx_c]).mean(axis=0)
            W += diff / (len(classes) - 1)
        if same.size > 0:
            diff_hit = np.abs(X[i] - X[same]).mean(axis=0)
            W -= diff_hit
    W /= max(1, len(idx_samples))
    return W

def reportar_top_features_4_6(scores, feature_names, topk=10, titulo=""):
    magnitudes = np.abs(scores)
    order = np.argsort(magnitudes)[::-1][:topk]
    print(f"\nTop-{topk} features {titulo}:")
    for rank, j in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_names[j]}  (score={scores[j]:.4f}, |score|={magnitudes[j]:.4f})")

# --- META 2 ---

# --- TAREFA 1: Data Augmentation ---

def filtrar_atividades_meta2(X, y):
    """
    (Helper Meta 2) Filtra X e y para manter apenas atividades 1-7.
    
    Parâmetros:
        X: matriz de features (n_amostras, n_features)
        y: vetor de rótulos (n_amostras,)
    
    Retorna:
        X_filtrado, y_filtrado
    """
    mask = y <= ATIVIDADE_META2_MAX
    return X[mask], y[mask]

def analisar_balanceamento_1_1(y):
    """
    (Tarefa 1.1) Analisa o equilíbrio das atividades 1 a 7.
    Nota: Esta função assume que y já foi filtrado para conter apenas atividades 1-7.
    """
    print("\n--- Tarefa 1.1: Análise de Balanceamento (Atividades 1-7) ---")
    
    mask = y <= ATIVIDADE_META2_MAX
    y_subset = y[mask]
    
    classes, counts = np.unique(y_subset, return_counts=True)
    
    print("Contagem de amostras por atividade:")
    total = 0
    for cls, count in zip(classes, counts):
        print(f"  Atividade {int(cls)}: {count} amostras")
        total += count
    
    print(f"Total de amostras (atividades 1-7): {total}")
        
    std_counts = np.std(counts)
    media_counts = np.mean(counts)
    cv = std_counts / media_counts 
    
    print(f"\nEstatísticas de balanceamento:")
    print(f"  Média: {media_counts:.1f}")
    print(f"  Desvio padrão: {std_counts:.1f}")
    print(f"  Coeficiente de variação: {cv:.3f}")
    
    if cv < 0.2: 
        print("-> O dataset parece razoavelmente balanceado.")
    else:
        print("-> O dataset NÃO está balanceado (diferenças significativas nas contagens).")
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribuição das Atividades (1-7)", fontsize=16, fontweight='bold')
    plt.xlabel("Atividade", fontsize=14)
    plt.ylabel("Número de Segmentos", fontsize=14)
    plt.xticks(classes)
    plt.grid(axis='y', alpha=0.5)
    
    for cls, count in zip(classes, counts):
        plt.text(cls, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("graphs/meta2_tarefa1_1_balanceamento.png", dpi=150)
    print("\nGráfico de balanceamento guardado em 'graphs/meta2_tarefa1_1_balanceamento.png'.")
    
    print("--- Fim Tarefa 1.1 ---")

def gerar_smote_1_2(X, y, atividade_alvo, K, k_neighbors=5, random_state=42):
    """
    (Tarefa 1.2) Implementação do SMOTE para gerar K novas amostras
    para uma dada atividade.
    
    Parâmetros:
        X: matriz de features (n_amostras, n_features) - apenas atividades 1-7
        y: vetor de rótulos (n_amostras,) - apenas atividades 1-7
        atividade_alvo: ID da atividade para augmentação (deve estar em ATIVIDADES_META2)
        K: número de amostras sintéticas a gerar
        k_neighbors: número de vizinhos a considerar
        random_state: seed para reprodutibilidade
    
    Retorna: 
        array numpy com as K amostras sintéticas (K, n_features)
    """
    np.random.seed(random_state)
    
    mask = y == atividade_alvo
    X_classe = X[mask]
    n_amostras = X_classe.shape[0]
    
    if n_amostras < 2:
        print(f"[Aviso] Amostras insuficientes ({n_amostras}) para SMOTE na atividade {atividade_alvo}.")
        return np.zeros((0, X.shape[1]))

    k_neighbors = min(k_neighbors, n_amostras - 1)
    if k_neighbors < 1:
        print(f"[Aviso] k_neighbors ajustado para {k_neighbors}, mas é insuficiente.")
        return np.zeros((0, X.shape[1]))

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(X_classe)
    
    amostras_sinteticas = []
    
    print(f"  A gerar {K} amostras sintéticas para atividade {atividade_alvo}...")
    
    for i in range(K):
        idx_base = np.random.randint(0, n_amostras)
        vetor_base = X_classe[idx_base]
        
        distancias, indices = nbrs.kneighbors(vetor_base.reshape(1, -1))
        vizinhos_indices = indices[0][1:]
        
        idx_vizinho = np.random.choice(vizinhos_indices)
        vetor_vizinho = X_classe[idx_vizinho]
        
        gap = np.random.random()
        diff = vetor_vizinho - vetor_base
        novo_vetor = vetor_base + (gap * diff)
        
        amostras_sinteticas.append(novo_vetor)
        
    return np.array(amostras_sinteticas)

def tarefa_1_3_visualizacao(base_dir="."):
    """
    (Tarefa 1.3) Gera e visualiza 3 novas amostras da atividade 4
    do participante 3.
    Nota: Filtra automaticamente para manter apenas atividades 1-7 (META 2).
    """
    print("\n--- Tarefa 1.3: Visualização SMOTE (Participante 3, Ativ 4) ---")
    
    dados_p3 = carregar_dados_participante(3, base_dir=base_dir)
    
    if dados_p3.size == 0:
        print("[Erro] Não foi possível carregar dados do participante 3.")
        return

    print("A extrair features do participante 3...")
    X_p3, y_p3, feat_names = construir_feature_set_4_2(dados_p3, fs=51.2)
    
    if X_p3.shape[0] == 0:
        print("[Erro] Nenhuma feature extraída para o participante 3.")
        return
    
    print(f"Features extraídas: {X_p3.shape}")
    
    mask_valid = y_p3 <= ATIVIDADE_META2_MAX
    X_p3 = X_p3[mask_valid]
    y_p3 = y_p3[mask_valid]
    
    print(f"Após filtro (ativ 1-{ATIVIDADE_META2_MAX}): {X_p3.shape}")

    atividade_alvo = 4
    K = 3
    print(f"\nA gerar {K} amostras sintéticas para a atividade {atividade_alvo}...")
    X_sintetico = gerar_smote_1_2(X_p3, y_p3, atividade_alvo, K=K, k_neighbors=5)
    
    if X_sintetico.shape[0] != K:
        print("[Erro] Falha ao gerar amostras sintéticas.")
        return
    
    print(f"Amostras sintéticas geradas com sucesso: {X_sintetico.shape}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1 = axes[0]
    
    classes_presentes = np.unique(y_p3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes_presentes)))
    
    for cls, color in zip(classes_presentes, colors):
        mask = y_p3 == cls
        label_text = f'Atividade {int(cls)}'
        if cls == atividade_alvo:
            label_text += ' (Original)'
        ax1.scatter(X_p3[mask, 0], X_p3[mask, 1], 
                   alpha=0.5, s=30, color=color, label=label_text)
    
    ax1.scatter(X_sintetico[:, 0], X_sintetico[:, 1], 
               color='red', marker='*', s=400, 
               edgecolors='black', linewidths=1.5,
               label=f'Sintético SMOTE (Ativ {atividade_alvo})', 
               zorder=10)
    
    ax1.set_xlabel(f"Feature 1: {feat_names[0]}", fontsize=12)
    ax1.set_ylabel(f"Feature 2: {feat_names[1]}", fontsize=12)
    ax1.set_title(f"SMOTE - Features 1 vs 2", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = axes[1]
    
    for cls, color in zip(classes_presentes, colors):
        mask = y_p3 == cls
        label_text = f'Atividade {int(cls)}'
        if cls == atividade_alvo:
            label_text += ' (Original)'
        ax2.scatter(X_p3[mask, 2], X_p3[mask, 3], 
                   alpha=0.5, s=30, color=color, label=label_text)
    
    ax2.scatter(X_sintetico[:, 2], X_sintetico[:, 3], 
               color='red', marker='*', s=400, 
               edgecolors='black', linewidths=1.5,
               label=f'Sintético SMOTE (Ativ {atividade_alvo})', 
               zorder=10)
    
    ax2.set_xlabel(f"Feature 3: {feat_names[2]}", fontsize=12)
    ax2.set_ylabel(f"Feature 4: {feat_names[3]}", fontsize=12)
    ax2.set_title(f"SMOTE - Features 3 vs 4", fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    filename = "graphs/meta2_tarefa1_3_smote_visualizacao.png"
    plt.savefig(filename, dpi=150)
    print(f"\nGráfico guardado em '{filename}'.")
    
    print("\n--- Estatísticas das Amostras Sintéticas ---")
    print(f"Forma: {X_sintetico.shape}")
    print(f"Primeiras 5 features da 1ª amostra sintética:")
    print(X_sintetico[0, :5])
    
    X_orig_ativ4 = X_p3[y_p3 == atividade_alvo]
    media_orig = X_orig_ativ4.mean(axis=0)
    print(f"\nMédia das originais (ativ {atividade_alvo}, primeiras 5 features):")
    print(media_orig[:5])
    
    print("--- Fim Tarefa 1.3 ---")


# --- TAREFA 2: Embeddings com Modelo Pré-treinado ---

def gerar_embeddings_dataset_2_1(segmentos, fs_original=51.2):
    """
    (Tarefa 2.1) Aplica as funções do embeddings_extractor.py para criar o dataset.
    """
    print(f"\n--- Tarefa 2.1: A gerar Embeddings para {len(segmentos)} segmentos ---")
    
    if not segmentos:
        print("[Aviso] Nenhum segmento para processar.")
        return np.array([])

    resampled_segments = []
    
    for i, seg in enumerate(segmentos):
        acc_data = seg['X'][:, 0:3]
        acc_30hz, _ = resample_to_30hz_5s(acc_data, fs_in_hz=fs_original)
        resampled_segments.append(acc_30hz)

    x_all = np.array(resampled_segments)
    x_all = np.transpose(x_all, (0, 2, 1))
    
    print(f"  Shape de entrada no modelo: {x_all.shape}")

    try:
        feature_encoder = load_model()
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o modelo (verifique a internet/PyTorch): {e}")
        return np.array([])

    embeddings_list = []
    batch_size = 32 
    
    with torch.no_grad(): 
        for i in range(0, x_all.shape[0], batch_size):
            batch_numpy = x_all[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_numpy).float().to("cpu")
            emb_batch = feature_encoder(batch_tensor)
            embeddings_list.append(emb_batch.cpu().numpy())
            
            if (i // batch_size) % 5 == 0:
                print(f"    Processado batch {i // batch_size}...")

    embeddings_final = np.concatenate(embeddings_list, axis=0)
    print(f"  Embeddings gerados com sucesso. Shape final: {embeddings_final.shape}")
    
    return embeddings_final


# --- TAREFA 3: Validação e Classificação ---

def split_within_subject_3_1(X, y, subjects, X_emb=None, train_r=0.6, val_r=0.2, random_state=42):
    """
    (Tarefa 3.1) Divide os dados em Train/Val/Test mantendo as proporções 
    DENTRO de cada sujeito. 
    """
    print(f"\n--- Tarefa 3.1: Split Within-Subject ({train_r*100:.0f}-{val_r*100:.0f}-{(1-train_r-val_r)*100:.0f}%) ---")
    
    np.random.seed(random_state)
    
    indices_train = []; indices_val = []; indices_test = []
    
    unique_subs = np.unique(subjects)
    print(f"Número de sujeitos únicos: {len(unique_subs)}")
    
    for s in unique_subs:
        idx_s = np.where(subjects == s)[0]
        np.random.shuffle(idx_s)
        
        n = len(idx_s)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        
        indices_train.extend(idx_s[:n_train])
        indices_val.extend(idx_s[n_train : n_train + n_val])
        indices_test.extend(idx_s[n_train + n_val:])
    
    idx_train = np.array(indices_train); idx_val = np.array(indices_val); idx_test = np.array(indices_test)
    
    data_split = {
        'X_train': X[idx_train], 'y_train': y[idx_train],
        'X_val':   X[idx_val],   'y_val':   y[idx_val],
        'X_test':  X[idx_test],  'y_test':  y[idx_test],
        'sub_train': subjects[idx_train],
        'sub_val':   subjects[idx_val],
        'sub_test':  subjects[idx_test]
    }
    
    if X_emb is not None:
        data_split['X_emb_train'] = X_emb[idx_train]
        data_split['X_emb_val']   = X_emb[idx_val]
        data_split['X_emb_test']  = X_emb[idx_test]
        print(f"  Embeddings também divididos.")
    
    print(f"  Train: {len(idx_train)} amostras | Val: {len(idx_val)} amostras | Test: {len(idx_test)} amostras")
    
    return data_split

def split_between_subject_3_2(X, y, subjects, X_emb=None, train_subs=9, val_subs=3, test_subs=3, random_state=42):
    """
    (Tarefa 3.2) Divide os dados em Train/Val/Test ao nível do SUJEITO.
    """
    print(f"\n--- Tarefa 3.2: Split Between-Subject ({train_subs} treinar, {val_subs} validar, {test_subs} testar) ---")
    
    np.random.seed(random_state)
    
    unique_subs = np.unique(subjects)
    n_total = len(unique_subs)
    
    np.random.shuffle(unique_subs)
    
    subs_train = unique_subs[:train_subs]
    subs_val   = unique_subs[train_subs : train_subs + val_subs]
    subs_test  = unique_subs[train_subs + val_subs :]
    
    print(f"  Sujeitos Treino: {subs_train}")
    print(f"  Sujeitos Validação: {subs_val}")
    print(f"  Sujeitos Teste: {subs_test}")
    
    mask_train = np.isin(subjects, subs_train)
    mask_val   = np.isin(subjects, subs_val)
    mask_test  = np.isin(subjects, subs_test)
    
    data_split = {
        'X_train': X[mask_train], 'y_train': y[mask_train],
        'X_val':   X[mask_val],   'y_val':   y[mask_val],
        'X_test':  X[mask_test],  'y_test':  y[mask_test],
        'sub_train': subjects[mask_train], 
        'sub_val':   subjects[mask_val],
        'sub_test':  subjects[mask_test]
    }
    
    if X_emb is not None:
        data_split['X_emb_train'] = X_emb[mask_train]
        data_split['X_emb_val']   = X_emb[mask_val]
        data_split['X_emb_test']  = X_emb[mask_test]
        
    print(f"  Train: {data_split['y_train'].shape[0]} amostras | Val: {data_split['y_val'].shape[0]} amostras | Test: {data_split['y_test'].shape[0]} amostras")
    
    return data_split

def processar_cenarios_3_4(X_train, X_val, X_test, y_train, y_val, y_test, top_k=15, pca_var=0.90):
    """
    (Tarefa 3.4 - Atualizada) Gera cenários e GUARDA OS LABELS (y) junto.
    """
    cenarios = {}

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm   = scaler.transform(X_val)
    X_test_norm  = scaler.transform(X_test)

    cenarios['all'] = {
        'X_train': X_train_norm, 'y_train': y_train,
        'X_val':   X_val_norm,   'y_val':   y_val,
        'X_test':  X_test_norm,  'y_test':  y_test,
        'scaler': scaler
    }

    print(f"  > A ajustar PCA (var={pca_var*100:.0f}%)...")
    pca = PCA(n_components=pca_var)
    pca.fit(X_train_norm)
    
    cenarios['pca'] = {
        'X_train': pca.transform(X_train_norm), 'y_train': y_train,
        'X_val':   pca.transform(X_val_norm),   'y_val':   y_val,
        'X_test':  pca.transform(X_test_norm),  'y_test':  y_test,
        'model': pca, 'scaler': scaler
    }

    print(f"  > A executar ReliefF (Top {top_k})...")
    scores = relieff_4_5(X_train_norm, y_train, n_neighbors=10, n_samples=2000)
    order = np.argsort(np.abs(scores))[::-1]
    top_indices = order[:top_k]
    
    cenarios['relief'] = {
        'X_train': X_train_norm[:, top_indices], 'y_train': y_train,
        'X_val':   X_val_norm[:, top_indices],   'y_val':   y_val,
        'X_test':  X_test_norm[:, top_indices],  'y_test':  y_test,
        'indices': top_indices, 'scaler': scaler
    }
    
    return cenarios

def pipeline_preparacao_3_4(split_data):
    """
    (Atualizada) Passa os y_train, y_val, y_test para o processamento.
    """
    resultados = {}
    
    y_tr = split_data['y_train']
    y_va = split_data['y_val']
    y_te = split_data['y_test']
    
    print("\n--- Tarefa 3.4: Preparar Cenários (FEATURES) ---")
    resultados['features'] = processar_cenarios_3_4(
        split_data['X_train'], split_data['X_val'], split_data['X_test'],
        y_tr, y_va, y_te  
    )
    
    if 'X_emb_train' in split_data:
        print("\n--- Tarefa 3.4: Preparar Cenários (EMBEDDINGS) ---")
        X_train_e = split_data['X_emb_train']
        X_val_e   = split_data['X_emb_val']
        X_test_e  = split_data['X_emb_test']
        
        if X_train_e.ndim > 2:
            X_train_e = X_train_e.squeeze()
            X_val_e   = X_val_e.squeeze()
            X_test_e  = X_test_e.squeeze()

        resultados['embeddings'] = processar_cenarios_3_4(
            X_train_e, X_val_e, X_test_e,
            y_tr, y_va, y_te
        )
        
    return resultados


# --- Tarefa 4.1: Implementação Manual do k-NN ---

class KNNClassifierCustom:
    """
    Implementação manual do k-Nearest Neighbors (k-NN).
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for i, row in enumerate(X_test):
            distances = np.sqrt(np.sum((self.X_train - row)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            try:
                moda_result = stats.mode(k_nearest_labels, keepdims=True)
                pred = moda_result.mode[0]
            except TypeError:
                pred = stats.mode(k_nearest_labels).mode[0]
            predictions.append(pred)
        return np.array(predictions)


def avaliar_resultados_4_2(y_true, y_pred, titulo="Resultados"):
    """
    (Tarefa 4.2) Calcula e apresenta métricas de classificação.
    """
    print(f"\n--- Relatório de Avaliação: {titulo} ---")
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n  Relatório por Classe:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm
    }
    
    return metrics

def plotar_matriz_confusao(cm, classes, titulo="Matriz de Confusão"):
    """
    (Auxiliar) Gera um heatmap bonito da matriz de confusão.
    """
    print("Seaborn not installed. Skipping confusion matrix plot.")

# --- Tarefa 5.1: Sintonização de Hiperparâmetros (k) ---

def sintonizar_k_5_1(X_train, y_train, X_val, y_val, k_lista=range(1, 20, 2)):
    """
    (Tarefa 5.1) Testa vários valores de k usando o conjunto de Validação.
    """
    best_k = -1
    best_score = -1.0
    historico = [] 
    
    for k in k_lista:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        preds_val = knn.predict(X_val)
        score = f1_score(y_val, preds_val, average='weighted', zero_division=0)
        
        historico.append((k, score))
        
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k, best_score, historico

# --- TAREFA 5.3: Testes Estatísticos ---
def run_statistical_comparison_5_3(X_feat, X_emb, y, n_splits=5):
    print(f"\n--- Tarefa 5.3: Teste Estatístico (Comparando Features vs Embeddings em {n_splits} splits) ---")
    if X_emb.ndim > 2: X_emb = X_emb.squeeze()
    
    scores_feat = []
    scores_emb = []
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    
    for i, (train_index, test_index) in enumerate(sss.split(X_feat, y)):
        y_tr, y_te = y[train_index], y[test_index]
        
        sc_f = StandardScaler().fit(X_feat[train_index])
        knn_f = KNeighborsClassifier(n_neighbors=5).fit(sc_f.transform(X_feat[train_index]), y_tr)
        acc_f = knn_f.score(sc_f.transform(X_feat[test_index]), y_te)
        scores_feat.append(acc_f)
        
        sc_e = StandardScaler().fit(X_emb[train_index])
        knn_e = KNeighborsClassifier(n_neighbors=5).fit(sc_e.transform(X_emb[train_index]), y_tr)
        acc_e = knn_e.score(sc_e.transform(X_emb[test_index]), y_te)
        scores_emb.append(acc_e)
        
        print(f"  Split {i+1}: Features={acc_f:.4f} vs Embeddings={acc_e:.4f}")
        
    stat, p_val = stats.ttest_rel(scores_feat, scores_emb)
    print(f"\n  Média Features: {np.mean(scores_feat):.4f}")
    print(f"  Média Embeddings: {np.mean(scores_emb):.4f}")
    print(f"  Paired t-test: stat={stat:.4f}, p-value={p_val:.4e}")
    if p_val < 0.05:
        print("  -> Diferença Estatisticamente Significativa!")
    else:
        print("  -> Sem diferença significativa.")

# --- TAREFA 6: Deployment ---
def classificar_atividade_6(raw_data_256_9, modelo_dict):
    seg = {'device': 0, 'activity': 0, 'X': raw_data_256_9}
    features, _ = extrair_features_janela_4_2(seg, fs=51.2)
    features = features.reshape(1, -1)
    
    if modelo_dict.get('scaler'): features = modelo_dict['scaler'].transform(features)
    
    if modelo_dict.get('pca'): features = modelo_dict['pca'].transform(features)
    elif modelo_dict.get('indices') is not None: features = features[:, modelo_dict['indices']]
        
    pred = modelo_dict['model'].predict(features)
    return pred[0]

# --- TAREFA 7.1: Bónus ---
def bonus_task_7_1(X_train, y_train, X_test, y_test):
    print(f"\n--- Tarefa 7.1 (Bonus): Comparação com Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Random Forest Accuracy: {acc:.4f}")
    return rf

# --- Função Principal (main)  ---
def main():
    """
    Função principal que orquestra a execução das tarefas da Meta 1 e Meta 2.
    """

    # --- PAINEL DE CONTROLO ---
    RUN_TASK_2_TEST = False
    RUN_TASK_3_1 = False
    RUN_TASK_3_2 = False
    RUN_TASK_3_4 = False
    RUN_TASK_3_5 = False
    RUN_TASK_3_7 = False
    RUN_TASK_3_7_1 = False
    RUN_TASK_4_1 = False
    RUN_TASK_4_2 = False
    RUN_TASK_4_3_4_4 = False
    RUN_TASK_4_5_4_6 = False
    RUN_META2_TASK_1_1 = False
    RUN_META2_TASK_1_3 = False
    RUN_META2_TASK_2_1 = True
    RUN_META2_TASK_3_1 = True
    RUN_META2_TASK_3_2 = True
    RUN_META2_TASK_3_4 = True
    RUN_META2_TASK_4_1 = False
    RUN_META2_TASK_4_2 = False
    RUN_META2_TASK_5_1 = True
    RUN_META2_TASK_5_3 = True
    RUN_META2_TASK_6 = True
    RUN_META2_TASK_7_1 = True

    # --- Execução ---

    if RUN_TASK_2_TEST:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        carregar_dados_participante(1, base_dir=script_dir)

    if any([RUN_TASK_3_1, RUN_TASK_3_2, RUN_TASK_3_4, RUN_TASK_3_5, RUN_TASK_3_7, RUN_TASK_3_7_1, RUN_TASK_4_1, RUN_TASK_4_2, RUN_TASK_4_3_4_4, RUN_TASK_4_5_4_6]):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dados_todos = carregar_dados_todos_participantes(base_dir=script_dir)
        if dados_todos.size == 0: return

        if any([RUN_TASK_3_1, RUN_TASK_3_2, RUN_TASK_3_4, RUN_TASK_3_5, RUN_TASK_4_1]):
            dados_transformados = get_dados_transformados(dados_todos)
            if RUN_TASK_3_1: plotar_boxplots_3_1(dados_transformados)
            if RUN_TASK_3_2: analisar_densidade_iqr_3_2(dados_transformados)
            if RUN_TASK_3_4: plotar_outliers_zscore_3_4(dados_transformados)
            if RUN_TASK_3_5: comparar_densidades_3_5(dados_transformados)
            if RUN_TASK_4_1: testar_significancia_medias_4_1(dados_transformados)

        if RUN_TASK_3_7: analisar_outliers_kmeans_3_7(dados_todos, n_clusters_lista=[2, 3, 5])
        if RUN_TASK_3_7_1: analisar_outliers_dbscan_3_7_1(dados_todos)

        if RUN_TASK_4_2 or RUN_TASK_4_3_4_4 or RUN_TASK_4_5_4_6:
            print("\nA construir feature set (Tarefa 4.2)...")
            X, y, feat_names = construir_feature_set_4_2(dados_todos, fs=51.2)
            if RUN_TASK_4_3_4_4: pca_pipeline_4_3_4_4(X)
            if RUN_TASK_4_5_4_6:
                Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
                fisher = fisher_score_multi_4_5(Xn, y)
                reportar_top_features_4_6(fisher, feat_names, topk=10, titulo="(Fisher)")

    FLAGS_META_2 = [RUN_META2_TASK_1_1, RUN_META2_TASK_1_3, RUN_META2_TASK_2_1, RUN_META2_TASK_3_1, RUN_META2_TASK_3_2, RUN_META2_TASK_3_4, RUN_META2_TASK_4_1, RUN_META2_TASK_4_2, RUN_META2_TASK_5_1, RUN_META2_TASK_5_3, RUN_META2_TASK_6, RUN_META2_TASK_7_1]

    if any(FLAGS_META_2):
        print("\n" + "="*60)
        print("=== META 2 (MÓDULO B): DATA AUGMENTATION & EMBEDDINGS ===")
        print("="*60)
        print(f"Nota: Módulo B considera APENAS atividades 1-{ATIVIDADE_META2_MAX}\n")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if 'dados_todos' not in locals():
            dados_todos = carregar_dados_todos_com_id(base_dir=script_dir)
        if dados_todos.size == 0: return

        print("\n--- Etapa 2: Segmentação ---")
        segmentos = segmentar_janelas_com_sujeito(dados_todos, fs=51.2, janela_s=5.0, overlap=0.5)
        y_todos = np.array([seg['activity'] for seg in segmentos], dtype=int)
        subjects_todos = np.array([seg['subject'] for seg in segmentos], dtype=int)
        
        print("\n--- Etapa 3: Filtragem ---")
        mask_mod_b = y_todos <= ATIVIDADE_META2_MAX
        segmentos_mod_b = [seg for i, seg in enumerate(segmentos) if mask_mod_b[i]]
        y_mod_b = y_todos[mask_mod_b]
        subjects_mod_b = subjects_todos[mask_mod_b]
        print(f"Segmentos após filtro: {len(segmentos_mod_b)}")

        if 'X_features' not in locals():
            print("\nExtraindo features manuais...")
            X_features_list = []
            for seg in segmentos_mod_b:
                fv, _ = extrair_features_janela_4_2(seg, fs=51.2)
                X_features_list.append(fv)
            X_features = np.vstack(X_features_list)

        if RUN_META2_TASK_1_1: analisar_balanceamento_1_1(y_mod_b)
        if RUN_META2_TASK_1_3: tarefa_1_3_visualizacao(base_dir=script_dir)
        
        if RUN_META2_TASK_2_1 or RUN_META2_TASK_5_3:
            if os.path.exists("meta2_embeddings_X.npy"):
                print("A carregar embeddings...")
                X_embeddings = np.load("meta2_embeddings_X.npy")
                if len(X_embeddings) != len(segmentos_mod_b):
                    X_embeddings = gerar_embeddings_dataset_2_1(segmentos_mod_b)
            else:
                X_embeddings = gerar_embeddings_dataset_2_1(segmentos_mod_b)
            if not os.path.exists("meta2_embeddings_X.npy") and X_embeddings.size > 0:
                np.save("meta2_embeddings_X.npy", X_embeddings)
                np.save("meta2_embeddings_y.npy", y_mod_b)

        if RUN_META2_TASK_3_1:
            X_emb_param = X_embeddings if 'X_embeddings' in locals() else None
            split_data = split_within_subject_3_1(X_features, y_mod_b, subjects_mod_b, X_emb=X_emb_param)
        
        if RUN_META2_TASK_3_2:
            X_emb_param = X_embeddings if 'X_embeddings' in locals() else None
            split_data_v2 = split_between_subject_3_2(X_features, y_mod_b, subjects_mod_b, X_emb=X_emb_param)

        estrategias_disponiveis = {}
        if RUN_META2_TASK_3_4:
            if 'split_data_v2' in locals():
                print("\nProcessando Between-Subject...")
                estrategias_disponiveis['Between-Subject'] = pipeline_preparacao_3_4(split_data_v2)
            if 'split_data' in locals():
                print("\nProcessando Within-Subject...")
                estrategias_disponiveis['Within-Subject'] = pipeline_preparacao_3_4(split_data)

        if RUN_META2_TASK_4_1 and estrategias_disponiveis:
            strat = list(estrategias_disponiveis.values())[0]
            print(f"Teste Manual k-NN: {KNNClassifierCustom(3).fit(strat['features']['all']['X_train'], strat['features']['all']['y_train']).predict(strat['features']['all']['X_test'][:5])}")

        if RUN_META2_TASK_4_2 and estrategias_disponiveis:
             strat = list(estrategias_disponiveis.values())[0]
             knn = KNeighborsClassifier(3).fit(strat['features']['all']['X_train'], strat['features']['all']['y_train'])
             avaliar_resultados_4_2(strat['features']['all']['y_test'][:100], knn.predict(strat['features']['all']['X_test'][:100]))

        melhores_ks = {}
        if RUN_META2_TASK_5_1 and estrategias_disponiveis:
            print("\n--- TAREFA 5: Tuning e Avaliação Final ---")
            for nome_strat, cenarios in estrategias_disponiveis.items():
                print(f"\n>>> Estratégia: {nome_strat}")
                melhores_ks[nome_strat] = {}
                for tipo in ['features', 'embeddings']:
                    if tipo not in cenarios: continue
                    melhores_ks[nome_strat][tipo] = {}
                    for proc in ['all', 'pca', 'relief']:
                        if proc not in cenarios[tipo]: continue
                        d = cenarios[tipo][proc]
                        bk, bs, _ = sintonizar_k_5_1(d['X_train'], d['y_train'], d['X_val'], d['y_val'])
                        melhores_ks[nome_strat][tipo][proc] = bk
                        print(f"    [{tipo}|{proc}] Melhor k={bk} (Val F1={bs:.2%})")
                        xf = np.vstack([d['X_train'], d['X_val']]); yf = np.concatenate([d['y_train'], d['y_val']])
                        knn = KNeighborsClassifier(n_neighbors=bk).fit(xf, yf)
                        preds = knn.predict(d['X_test'])
                        print(f"      TESTE Accuracy: {accuracy_score(d['y_test'], preds):.2%}")
                        print(classification_report(d['y_test'], preds, zero_division=0))

        if RUN_META2_TASK_5_3:
             if 'X_features' in locals() and 'X_embeddings' in locals():
                  run_statistical_comparison_5_3(X_features, X_embeddings, y_mod_b)

        if RUN_META2_TASK_6 and melhores_ks:
             try:
                 strat_name = 'Between-Subject' if 'Between-Subject' in estrategias_disponiveis else 'Within-Subject'
                 d = estrategias_disponiveis[strat_name]['features']['all']
                 k = melhores_ks[strat_name]['features']['all']
                 print(f"\n--- Tarefa 6: Deployment (k={k}) ---")
                 mod = {'scaler': d['scaler'], 'model': KNeighborsClassifier(n_neighbors=k).fit(d['X_train'], d['y_train'])}
                 print(f"  Predição dummy: {classificar_atividade_6(np.random.randn(256,9), mod)}")
             except:
                 pass

        if RUN_META2_TASK_7_1 and estrategias_disponiveis:
             d = list(estrategias_disponiveis.values())[0]['features']['all']
             bonus_task_7_1(d['X_train'], d['y_train'], d['X_test'], d['y_test'])

    print("\n=== Execução Concluída ===")

if __name__ == "__main__":
    main()
