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
from mpl_toolkits.mplot3d import Axes3D  # Necessário para plots 3D
from sklearn.cluster import DBSCAN      # Apenas para o bónus 3.7.1
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch

# --- Constantes Globais (Baseado no PDF) ---
# Mapeamento 1-para-1 com o PDF (1-based-indexing) para 0-based-indexing
COL_DEVICE_ID = 0  # Coluna 1
COL_ACC_X = 1      # Coluna 2
COL_ACC_Y = 2      # Coluna 3
COL_ACC_Z = 3      # Coluna 4
COL_GYRO_X = 4     # Coluna 5
COL_GYRO_Y = 5     # Coluna 6
COL_GYRO_Z = 6     # Coluna 7
COL_MAG_X = 7      # Coluna 8
COL_MAG_Y = 8      # Coluna 9
COL_MAG_Z = 9      # Coluna 10
COL_TIMESTAMP = 10 # Coluna 11
COL_ACTIVITY = 11  # Coluna 12
COL_PARTICIPANT = 12 # Coluna 13 - ID do participante - Necessário para Meta 2 ex 3!

# Grupos de colunas para cálculo dos módulos
MODULO_INDICES = [
    (COL_ACC_X, COL_ACC_Y, COL_ACC_Z),  # Aceleração
    (COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z), # Giroscópio
    (COL_MAG_X, COL_MAG_Y, COL_MAG_Z)  # Magnetómetro
]

# Labels para gráficos
VAR_LABELS = [
    "Módulo Acelerômetro",
    "Módulo Giroscópio",
    "Módulo Magnetômetro"
]
SENSOR_LABELS = {
    1: "Pulso Esquerdo",  # ID 1
    2: "Pulso Direito",   # ID 2
    3: "Peito",           # ID 3
    4: "Perna Sup. Direita", # ID 4
    5: "Perna Inf. Esquerda" # ID 5
}

# Constante para Meta 2: Atividades válidas (1 a 7)
ATIVIDADES_META2 = [1, 2, 3, 4, 5, 6, 7]
ATIVIDADE_META2_MAX = 7


# --- Funções baseadas no embeddings_extractor.py ---

def load_model():
    """
    (Do ficheiro embeddings_extractor.py)
    Carrega o modelo HARNet5 do repositório GitHub e extrai o feature_encoder.
    """
    print("A carregar modelo HARNet5 (necessita de internet na 1ª execução)...")
    repo = 'OxWearables/ssl-wearables'
    # class_num não é usado para extração, mas é argumento obrigatório
    model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
    model.eval()

    # Extrair apenas o extrator de features (camadas convolucionais)
    feature_encoder = model.feature_extractor
    feature_encoder.to("cpu")
    feature_encoder.eval()
    return feature_encoder

def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    (Do ficheiro embeddings_extractor.py)
    Reamostra um segmento de acelerómetro para 30Hz.
    Retorna: (sinal_reamostrado, nova_frequencia)
    """
    fs_target = 30.0
    win_size = 5 # segundos
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)

    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    return acc_resampled, fs_target

# -----------------------------------

# ///// META 1 /////

# --- Tarefa 2: Carregamento de Dados  ---

def carregar_dados_participante(num_participante, base_dir="."):
    """
    (Tarefa 2) Lê todos os 5 ficheiros CSV do participante e devolve um array NumPy.
    """
    part_dir = os.path.join(base_dir, f"part{num_participante}")
    dados = []

    for i in range(1, 6): # Dispositivos 1 a 5
        caminho_ficheiro = os.path.join(part_dir, f"part{num_participante}dev{i}.csv")
        try:
            with open(caminho_ficheiro, newline='') as csvfile:
                leitor = csv.reader(csvfile)
                # O PDF não menciona cabeçalho, mas é boa prática saltar
                # next(leitor, None)
                for linha in leitor:
                    try:
                        # Tenta converter toda a linha para float
                        dados.append([float(x) for x in linha])
                    except ValueError:
                        # Ignora linhas que não podem ser convertidas (ex: cabeçalhos)
                        continue
        except FileNotFoundError:
            print(f"[Aviso] Ficheiro não encontrado: {caminho_ficheiro}")
        except Exception as e:
            print(f"[Erro] Problema ao ler {caminho_ficheiro}: {e}")

    if not dados:
        return np.array([]) # Retorna array vazio se nada foi carregado

    return np.array(dados)

def carregar_dados_todos_participantes(base_dir="."):
    """
    (Helper para Tarefa 3.1) Carrega dados de TODOS os 15 participantes.

    """
    dados_todos = []
    # O dataset tem 15 participantes, com pastas part0 a part14.
    for part_num in range(15):
        print(f"A carregar participante {part_num}...")
        dados_participante = carregar_dados_participante(part_num, base_dir)
        if dados_participante.size > 0:
            dados_todos.append(dados_participante)

    if not dados_todos:
        print("[Erro Fatal] Nenhum dado carregado. Verifique o caminho `base_dir`.")
        return np.array([])

    return np.concatenate(dados_todos, axis=0)

def carregar_dados_todos_com_id(base_dir="."):
    """
    (Helper para Tarefa 3 Meta 2) Carrega dados de TODOS os 15 participantes
    e ADICIONA uma coluna com o ID do sujeito.
    """
    dados_todos = []
    for part_num in range(15):
        print(f"A carregar participante {part_num}...")
        dados_p = carregar_dados_participante(part_num, base_dir)
        if dados_p.size > 0:
            # Criar coluna com o ID (part_num) repetido
            col_id = np.full((dados_p.shape[0], 1), part_num)
            # Juntar aos dados existentes
            dados_p = np.hstack([dados_p, col_id])
            dados_todos.append(dados_p)
    
    if not dados_todos:
        print("[Erro Fatal] Nenhum dado carregado. Verifique o caminho `base_dir`.")
        return np.array([])
    
    return np.concatenate(dados_todos, axis=0)

# --- Tarefa 3 (Intro): Preparação de Dados ---

def calcular_modulo_vetor(dados, col_x, col_y, col_z):
    """
    Calcula o módulo (norma Euclidiana) de um vetor 3D.
    Fórmula: ||t|| = sqrt(tx^2 + ty^2 + tz^2)
    """
    # Seleciona as colunas [x, y, z]
    vetor = dados[:, [col_x, col_y, col_z]]
    # axis=1 calcula a norma para cada *linha*
    modulo = np.linalg.norm(vetor, axis=1)
    return modulo

def get_dados_transformados(dados):
    """
    (Helper) Cria um novo dataset estruturado com os módulos calculados.
    Isto evita recálculos e simplifica o código.
    """
    print("A calcular módulos (dataset transformado)...")
    # Estrutura: [ID_Sensor, ID_Atividade, Mod_Acel, Mod_Gyro, Mod_Mag]
    # Isto é uma "Feature Matrix" inicial.

    # Extrai colunas de ID e Atividade
    ids_sensores = dados[:, COL_DEVICE_ID]
    ids_atividades = dados[:, COL_ACTIVITY]

    # Prepara um array para os 3 módulos
    modulos = np.zeros((dados.shape[0], 3))

    for i, (cols) in enumerate(MODULO_INDICES):
        modulos[:, i] = calcular_modulo_vetor(dados, cols[0], cols[1], cols[2])

    # Empilha horizontalmente (coluna a coluna)
    # [ids_sensores, ids_atividades, mod_acel, mod_gyro, mod_mag]
    dados_transformados = np.stack([
        ids_sensores,
        ids_atividades,
        modulos[:, 0],
        modulos[:, 1],
        modulos[:, 2]
    ], axis=1)

    return dados_transformados

# --- Tarefa 3.1: Boxplot de Atividades  ---

def plotar_boxplots_3_1(dados_transformados):
    """
    (Tarefa 3.1) Gera 15 boxplots (3 vars x 5 sensores).
    Eixo X: Atividade (1-16)
    Eixo Y: Módulo da variável transformada
    """
    print("A gerar gráficos para Tarefa 3.1...")

    # 3 linhas (variáveis) x 5 colunas (sensores)
    fig, axes = plt.subplots(3, 5, figsize=(24, 15), sharex=True)
    fig.suptitle("Tarefa 3.1: Módulos por Atividade e Dispositivo (Todos os Sujeitos)",
                 fontsize=20, fontweight='bold')

    for var_idx in range(3): # 0=Acel, 1=Gyro, 2=Mag
        for sensor_id in range(1, 6): # 1 a 5

            ax = axes[var_idx, sensor_id-1] # -1 para 0-index do axes

            # 1. Filtrar dados para este sensor
            sensor_mask = dados_transformados[:, 0] == sensor_id
            dados_sensor = dados_transformados[sensor_mask]

            # 2. Preparar dados para o boxplot
            #    dados_box = [ lista_de_valores_ativ_1, lista_de_valores_ativ_2, ... ]
            dados_box = []
            for a in range(1, 17): # Atividades 1 a 16
                activity_mask = dados_sensor[:, 1] == a
                dados_ativ = dados_sensor[activity_mask, var_idx + 2] # +2 para aceder às cols de módulo
                dados_box.append(dados_ativ)

            # 3. Plotar
            # O `tick_labels=` define os ticks do eixo X (Matplotlib>=3.9)
            ax.boxplot(dados_box, tick_labels=[str(a) for a in range(1, 17)])

            # 4. Legendas
            if var_idx == 0:
                ax.set_title(SENSOR_LABELS[sensor_id], fontsize=14, fontweight='bold')
            if sensor_id == 1:
                ax.set_ylabel(VAR_LABELS[var_idx], fontsize=14, fontweight='bold')

            ax.set_xlabel("Atividade")
            ax.tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("graphs/meta1_tarefa_3_1_boxplots.png")
    print("Gráfico 'graphs/meta1_tarefa_3_1_boxplots.png' guardado.")
    # plt.show() # Descomentar para mostrar interativamente

# --- Tarefa 3.2: Análise Densidade de Outliers (IQR)  ---

def analisar_densidade_iqr_3_2(dados_transformados):
    """
    (Tarefa 3.2) Analisa e comenta a densidade de outliers.
    Usa método IQR (Tukey).
    Filtra apenas para o Pulso Direito (ID 2).
    """
    print("\n--- Início Tarefa 3.2: Densidade de Outliers (IQR - Pulso Direito) ---")

    sensor_id_foco = 2 # Pulso Direito

    # 1. Filtrar dados para o sensor
    sensor_mask = dados_transformados[:, 0] == sensor_id_foco
    dados_sensor = dados_transformados[sensor_mask]

    print(f"Analisando Sensor: {SENSOR_LABELS[sensor_id_foco]}\n")
    print("| Variável              | Atividade | n_total (nr) | n_outliers (no) | Densidade (d) |")
    print("|-----------------------|-----------|--------------|-----------------|---------------|")

    for var_idx in range(3): # Acel, Gyro, Mag
        var_label = VAR_LABELS[var_idx]

        for a in range(1, 17): # Atividades 1 a 16

            # 2. Filtrar dados para a atividade
            activity_mask = dados_sensor[:, 1] == a
            amostras = dados_sensor[activity_mask, var_idx + 2] # +2 para col de módulo

            n_r = len(amostras) # nr = número total de pontos
            if n_r == 0:
                continue # Pula atividades sem dados

            # 3. Aplicar método IQR (Tukey)
            q1 = np.percentile(amostras, 25)
            q3 = np.percentile(amostras, 75)
            iqr = q3 - q1

            limite_inf = q1 - (1.5 * iqr)
            limite_sup = q3 + (1.5 * iqr)

            # 4. Contar outliers (no)
            outliers_mask = (amostras < limite_inf) | (amostras > limite_sup)
            n_o = np.sum(outliers_mask) # no = número de outliers

            # 5. Calcular densidade (d)
            densidade = (n_o / n_r) * 100

            print(f"| {var_label:21} | {a:9} | {n_r:12} | {n_o:15} | {densidade:13.2f}% |")

    print("\n--- Fim Tarefa 3.2 ---")

# --- Tarefa 3.3: Rotina Z-Score  ---

def identificar_outliers_zscore_3_3(amostras, k):
    """
    (Tarefa 3.3) Identifica outliers usando Z-Score para um k variável.
    """
    if amostras.size == 0:
        return np.array([], dtype=bool)

    media = np.mean(amostras)
    std = np.std(amostras)

    # Evitar divisão por zero se o desvio padrão for 0
    if std == 0:
        return np.zeros(amostras.shape, dtype=bool)

    z_scores = (amostras - media) / std

    # Retorna uma máscara booleana (True onde for outlier)
    return np.abs(z_scores) > k

# --- Tarefa 3.4: Plots Z-Score  ---

def plotar_outliers_zscore_3_4(dados_transformados):
    """
    (Tarefa 3.4) Gera plots análogos a 3.1, mas usando Z-Score.
    Outliers a vermelho, inliers a azul.
    Testa k = 3, 3.5 e 4.
    """
    print("A gerar gráficos para Tarefa 3.4...")

    k_valores = [3, 3.5, 4]

    for k in k_valores:
        print(f"  A gerar para k={k}...")

        fig, axes = plt.subplots(3, 5, figsize=(24, 15), sharex=True)
        fig.suptitle(f"Tarefa 3.4: Outliers Z-Score (k={k})",
                     fontsize=20, fontweight='bold')

        for var_idx in range(3): # Acel, Gyro, Mag
            for sensor_id in range(1, 6): # 1 a 5

                ax = axes[var_idx, sensor_id-1]

                # Loop para plotar cada atividade
                for a in range(1, 17): # Atividades 1 a 16

                    # 1. Filtrar dados
                    sensor_mask = dados_transformados[:, 0] == sensor_id
                    activity_mask = dados_transformados[:, 1] == a
                    # Máscara combinada
                    mask = sensor_mask & activity_mask
                    amostras = dados_transformados[mask, var_idx + 2]

                    if amostras.size == 0:
                        continue

                    # 2. Identificar outliers
                    outlier_mask = identificar_outliers_zscore_3_3(amostras, k)
                    inlier_mask = ~outlier_mask

                    inliers = amostras[inlier_mask]
                    outliers = amostras[outlier_mask]

                    # 3. Plotar (com "jitter" horizontal para visualização)
                    x_base = a # Posição X
                    # Jitter: adiciona ruído gaussiano para espalhar os pontos
                    x_inliers = np.random.normal(x_base, 0.1, size=inliers.size)
                    x_outliers = np.random.normal(x_base, 0.1, size=outliers.size)

                    ax.scatter(x_inliers, inliers, color='blue', alpha=0.3, s=5, label="Inlier" if a==1 else "")
                    ax.scatter(x_outliers, outliers, color='red', alpha=1.0, s=10, label="Outlier" if a==1 else "")

                # 4. Legendas
                if var_idx == 0:
                    ax.set_title(SENSOR_LABELS[sensor_id], fontsize=14, fontweight='bold')
                if sensor_id == 1:
                    ax.set_ylabel(VAR_LABELS[var_idx], fontsize=14, fontweight='bold')
                if var_idx == 0 and sensor_id == 4:
                    ax.legend() # Adiciona legenda a um dos gráficos

                ax.set_xlabel("Atividade")
                ax.set_xticks(range(1, 17))
                ax.set_xticklabels([str(a) for a in range(1, 17)], rotation=90)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f"graphs/meta1_tarefa_3_4_zscore_k{k}.png")
        print(f"Gráfico 'graphs/meta1_tarefa_3_4_zscore_k{k}.png' guardado.")
        # plt.show()

# --- Tarefa 3.5: Comparação IQR vs Z-Score  ---

def comparar_densidades_3_5(dados_transformados):
    """
    (Tarefa 3.5) Compara e discute resultados de 3.1 (IQR) e 3.4 (Z-Score).
    Foco: Apenas sensores do pulso direito (ID 2).
    """
    print("\n--- Início Tarefa 3.5: Comparação Densidades (IQR vs Z-Score) - Pulso Direito ---")

    sensor_id_foco = 2 # Pulso Direito
    k_valores = [3, 3.5, 4]

    # 1. Filtrar dados
    sensor_mask = dados_transformados[:, 0] == sensor_id_foco
    dados_sensor = dados_transformados[sensor_mask]

    print(f"Analisando Sensor: {SENSOR_LABELS[sensor_id_foco]}\n")
    print("| Variável              | Ativ | n_total | Dens. IQR (%) | Dens. Z(k=3) (%) | Dens. Z(k=3.5) (%) | Dens. Z(k=4) (%) |")
    print("|-----------------------|------|---------|---------------|------------------|--------------------|------------------|")

    for var_idx in range(3): # Acel, Gyro, Mag
        var_label = VAR_LABELS[var_idx]

        for a in range(1, 17): # Atividades 1 a 16

            activity_mask = dados_sensor[:, 1] == a
            amostras = dados_sensor[activity_mask, var_idx + 2]

            n_r = len(amostras)
            if n_r == 0:
                continue

            # 2. Calcular Densidade IQR (como em 3.2)
            q1 = np.percentile(amostras, 25)
            q3 = np.percentile(amostras, 75)
            iqr = q3 - q1
            limite_inf = q1 - (1.5 * iqr)
            limite_sup = q3 + (1.5 * iqr)
            n_o_iqr = np.sum((amostras < limite_inf) | (amostras > limite_sup))
            d_iqr = (n_o_iqr / n_r) * 100

            # 3. Calcular Densidades Z-Score
            densidades_z = []
            for k in k_valores:
                outlier_mask_z = identificar_outliers_zscore_3_3(amostras, k)
                n_o_z = np.sum(outlier_mask_z)
                d_z = (n_o_z / n_r) * 100
                densidades_z.append(d_z)

            print(f"| {var_label:21} | {a:4} | {n_r:7} | {d_iqr:13.2f} | {densidades_z[0]:16.2f} | {densidades_z[1]:18.2f} | {densidades_z[2]:16.2f} |")


    print("--- Fim Tarefa 3.5 ---")

# --- Tarefa 3.6: Rotina K-Means  ---

def kmeans_3_6(X, n_clusters, max_iter=100, tol=1e-4, random_state=42):
    """
    (Tarefa 3.6) Implementa o algoritmo k-means.
    """
    # 1. Inicialização reprodutível
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], n_clusters, replace=False)
    centroides = X[indices]

    for _ in range(max_iter):
        # 2. Atribuição: Calcula distâncias de *todos* os pontos a *todos* os centroides
        #    Resultado é (n_amostras, n_clusters)
        distancias = np.sqrt(((X[:, np.newaxis] - centroides) ** 2).sum(axis=2))

        #    Encontra o índice do centroide mais próximo (axis=1)
        labels = np.argmin(distancias, axis=1)

        # 3. Atualização: Calcula o novo centroide (média) para cada cluster
        novos_centroides = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # 4. Verificação de Convergência
        #    Se a mudança nos centroides for muito pequena (abaixo da tolerância 'tol')
        if np.all(np.linalg.norm(novos_centroides - centroides, axis=1) < tol):
            break

        centroides = novos_centroides

    return centroides, labels

# --- Tarefa 3.7: Outliers com K-Means e Plot 3D  ---

def analisar_outliers_kmeans_3_7(dados, n_clusters_lista):
    """
    (Tarefa 3.7) Determina outliers usando k-means no espaço original (x,y,z).
    Gera plots 3D.
    """
    print(f"\n--- Início Tarefa 3.7: Outliers K-Means (n_clusters={n_clusters_lista}) ---")

    # Vamos focar-nos numa combinação para não gerar 16*5*3 gráficos
    # Ex: Aceleração (var 0), Pulso Direito (sensor 2), Atividade 4 (Walk)
    var_idx_foco = 0 # Aceleração
    sensor_id_foco = 2 # Pulso Direito
    atividade_foco = 4 # Walk

    cols_xyz = MODULO_INDICES[var_idx_foco]

    # 1. Filtrar dados
    sensor_mask = dados[:, COL_DEVICE_ID] == sensor_id_foco
    activity_mask = dados[:, COL_ACTIVITY] == atividade_foco
    mask = sensor_mask & activity_mask

    # X é o nosso dataset 3D (n_amostras, 3)
    X = dados[mask][:, [cols_xyz[0], cols_xyz[1], cols_xyz[2]]]

    if X.shape[0] < max(n_clusters_lista):
        print(f"Dados insuficientes para a atividade {atividade_foco}. A saltar.")
        return

    for n_clusters in n_clusters_lista:
        print(f"  A analisar k-means com n_clusters={n_clusters}...")

        # 2. Correr K-Means
        try:
            centroides, labels = kmeans_3_6(X, n_clusters, random_state=42)
        except ValueError as e:
            print(f"    Erro no K-Means (provavelmente cluster vazio): {e}")
            continue

        # 3. Estratégia de Outlier:
        #    Calcula a distância de cada ponto ao seu centroide
        distancias_ao_centroide = np.linalg.norm(X - centroides[labels], axis=1)

        #    Define outliers como pontos no 99º percentil de distância
        #    (i.e., o 1% de pontos mais distantes dos seus centroides)
        limite_dist = np.percentile(distancias_ao_centroide, 99)
        outlier_mask = distancias_ao_centroide > limite_dist
        inlier_mask = ~outlier_mask

        n_o = np.sum(outlier_mask)
        print(f"    Encontrados {n_o} outliers ({n_o/X.shape[0]*100:.2f}%)")

        # 4. Plot 3D (clusters coloridos distintamente)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Inliers coloridos por cluster
        sc = ax.scatter(
            X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2],
            c=labels[inlier_mask], cmap='tab10', alpha=0.6, s=10, label="Inliers (Clusters)"
        )
        # Outliers destacados
        ax.scatter(
            X[outlier_mask, 0], X[outlier_mask, 1], X[outlier_mask, 2],
            c='red', s=50, label="Outliers"
        )
        # Centroides
        ax.scatter(
            centroides[:, 0], centroides[:, 1], centroides[:, 2],
            c='black', s=200, marker='X', label="Centroides"
        )

        # Rótulos mais precisos (são componentes XYZ, não módulo)
        ax.set_xlabel("Acelerômetro X")
        ax.set_ylabel("Acelerômetro Y")
        ax.set_zlabel("Acelerômetro Z")
        ax.set_title(
            f"Tarefa 3.7: K-Means (k={n_clusters}) - Sensor '{SENSOR_LABELS[sensor_id_foco]}' - Atividade {atividade_foco}"
        )
        ax.legend()
        plt.savefig(f"graphs/meta1_tarefa_3_7_kmeans_k{n_clusters}.png")
        print(f"Gráfico 'graphs/meta1_tarefa_3_7_kmeans_k{n_clusters}.png' guardado.")

        # 5. Plot adicional: clusters sem outliers (para melhor visualização)
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(
            X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2],
            c=labels[inlier_mask], cmap='tab10', alpha=0.8, s=12, label="Inliers (Clusters)"
        )
        ax2.scatter(
            centroides[:, 0], centroides[:, 1], centroides[:, 2],
            c='black', s=200, marker='X', label="Centroides"
        )
        ax2.set_xlabel("Acelerômetro X")
        ax2.set_ylabel("Acelerômetro Y")
        ax2.set_zlabel("Acelerômetro Z")
        ax2.set_title(
            f"Tarefa 3.7: K-Means (k={n_clusters}) - SEM Outliers - Sensor '{SENSOR_LABELS[sensor_id_foco]}' - Atividade {atividade_foco}"
        )
        ax2.legend()
        plt.savefig(f"graphs/meta1_tarefa_3_7_kmeans_k{n_clusters}_sem_outliers.png")
        print(f"Gráfico 'graphs/meta1_tarefa_3_7_kmeans_k{n_clusters}_sem_outliers.png' guardado.")
        # plt.show()

    
    print("--- Fim Tarefa 3.7 ---")

# --- Tarefa 3.7.1 (Bónus): Outliers com DBSCAN  ---

def analisar_outliers_dbscan_3_7_1(dados):
    """
    (Tarefa 3.7.1) Determina outliers usando DBSCAN.
    """
    print(f"\n--- Início Tarefa 3.7.1 (Bónus): Outliers DBSCAN ---")

    # Usar a mesma fatia de dados da Tarefa 3.7 para comparação
    var_idx_foco = 0 # Aceleração
    sensor_id_foco = 2 # Pulso Direito
    atividade_foco = 4 # Walk

    cols_xyz = MODULO_INDICES[var_idx_foco]

    sensor_mask = dados[:, COL_DEVICE_ID] == sensor_id_foco
    activity_mask = dados[:, COL_ACTIVITY] == atividade_foco
    mask = sensor_mask & activity_mask
    X = dados[mask][:, [cols_xyz[0], cols_xyz[1], cols_xyz[2]]]

    if X.shape[0] < 50:
        print("Dados insuficientes. A saltar.")
        return

    # 1. Correr DBSCAN (eps heurístico via 5-NN)
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5).fit(X)
        dists, _ = nn.kneighbors(X)
        # distância ao 5º vizinho; usar percentil 95 como eps
        eps = np.percentile(dists[:, -1], 95)
        if eps <= 0:
            eps = 0.5
    except Exception:
        eps = 0.5
    db = DBSCAN(eps=eps, min_samples=10).fit(X)

    # 2. Outliers são pontos com label = -1
    outlier_mask = db.labels_ == -1
    inlier_mask = ~outlier_mask

    n_o = np.sum(outlier_mask)
    print(f"  DBSCAN (eps={eps:.3f}, min=10) encontrou {n_o} outliers ({n_o/X.shape[0]*100:.2f}%)")

    # 3. Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Inliers (coloridos por cluster)
    ax.scatter(
        X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2],
        c=db.labels_[inlier_mask], cmap='tab10', alpha=0.6, s=10, label="Inliers (Clusters)"
    )
    # Outliers
    ax.scatter(
        X[outlier_mask, 0], X[outlier_mask, 1], X[outlier_mask, 2],
        c='red', s=50, label="Outliers (Ruído)"
    )

    ax.set_xlabel("Acelerômetro X")
    ax.set_ylabel("Acelerômetro Y")
    ax.set_zlabel("Acelerômetro Z")
    ax.set_title(f"Tarefa 3.7.1: DBSCAN - Sensor '{SENSOR_LABELS[sensor_id_foco]}' - Atividade {atividade_foco}")
    ax.legend()
    plt.savefig("graphs/meta1_tarefa_3_7_1_dbscan.png")
    print("Gráfico 'graphs/meta1_tarefa_3_7_1_dbscan.png' guardado.")

    # Plot adicional: clusters sem outliers
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(
        X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2],
        c=db.labels_[inlier_mask], cmap='tab10', alpha=0.8, s=12, label="Inliers (Clusters)"
    )
    ax2.set_xlabel("Acelerômetro X")
    ax2.set_ylabel("Acelerômetro Y")
    ax2.set_zlabel("Acelerômetro Z")
    ax2.set_title(f"Tarefa 3.7.1: DBSCAN - SEM Outliers - Sensor '{SENSOR_LABELS[sensor_id_foco]}' - Atividade {atividade_foco}")
    ax2.legend()
    plt.savefig("graphs/meta1_tarefa_3_7_1_dbscan_sem_outliers.png")
    print("Gráfico 'graphs/meta1_tarefa_3_7_1_dbscan_sem_outliers.png' guardado.")
    # plt.show()
    print("--- Fim Tarefa 3.7.1 ---")


# --- Tarefa 4: Extração de Informação Característica ---

# 4.1 Testes de normalidade e significância dos valores médios (por atividade)
def testar_significancia_medias_4_1(dados_transformados):
    """
    Para cada variável transformada (módulos: acel, gyro, mag) e para cada sensor,
    testa normalidade por atividade (KS sobre Z) e, consoante resultado global,
    usa ANOVA (normal) ou Kruskal-Wallis (não-normal) para testar diferença de médias
    entre as 16 atividades.
    """
    print("\n--- Início Tarefa 4.1: Testes de Normalidade e Significância (médias por atividade) ---")

    resultados_normalidade = {}

    for sensor_id in range(1, 6):
        sensor_mask = dados_transformados[:, 0] == sensor_id
        dados_sensor = dados_transformados[sensor_mask]
        print(f"\nSensor: {SENSOR_LABELS[sensor_id]}")

        for var_idx in range(3):
            var_label = VAR_LABELS[var_idx]
            series_por_atividade = []
            normal_flags = []
            for a in range(1, 17):
                vals = dados_sensor[dados_sensor[:, 1] == a, var_idx + 2]
                if vals.size < 8:
                    # amostra demasiado pequena; marcar como não-normal por prudência
                    normal_flags.append(False)
                    continue
                # Teste de normalidade de D'Agostino
                try:
                    k2_stat, p_norm = stats.normaltest(vals)
                except Exception:
                    # fallback conservador
                    p_norm = 0.0
                normal_flags.append(p_norm > 0.05)
                series_por_atividade.append(vals)

            todos_normais = all(normal_flags) and len(series_por_atividade) >= 2

            if todos_normais:
                # One-way ANOVA
                f_stat, p_val = stats.f_oneway(*series_por_atividade)
                metodo = "ANOVA"
            else:
                # Kruskal-Wallis (não-paramétrico)
                try:
                    h_stat, p_val = stats.kruskal(*series_por_atividade)
                except ValueError:
                    p_val = np.nan
                metodo = "Kruskal-Wallis"

            print(f"  {var_label:20} -> teste={metodo:16} p={p_val:.3e}")
            resultados_normalidade[(sensor_id, var_idx)] = (metodo, p_val)

    print("--- Fim Tarefa 4.1 ---\n")


# 4.2 Janelação (5s, 50% overlap) e extração de features temporais e espectrais
def _windows_indices_4_2(n_samples, win_size, hop_size):
    for start in range(0, n_samples - win_size + 1, hop_size):
        yield start, start + win_size


def segmentar_janelas_puras_4_2(dados_raw, fs=51.2, janela_s=5.0, overlap=0.5):
    """
    Segmenta dados por dispositivo e atividade em janelas de 5s com 50% overlap.
    Descarta janelas que cruzam atividades (mantém apenas segmentos 'puros').

    Retorna: lista de dicts com chaves: 'device', 'activity', 'X' (matriz [N, 9])
    onde colunas 0-2: acc xyz, 3-5: gyro xyz, 6-8: mag xyz
    """
    win_size = int(janela_s * fs)
    hop_size = int(win_size * (1.0 - overlap))
    segmentos = []

    for sensor_id in range(1, 6):
        mask_dev = dados_raw[:, COL_DEVICE_ID] == sensor_id
        dados_dev = dados_raw[mask_dev]
        if dados_dev.size == 0:
            continue

        # ordenar por timestamp (segurança)
        idx_sort = np.argsort(dados_dev[:, COL_TIMESTAMP])
        dados_dev = dados_dev[idx_sort]

        # trabalhar por blocos de atividade constante
        atividades = dados_dev[:, COL_ACTIVITY].astype(int)
        change_idx = np.where(np.diff(atividades) != 0)[0] + 1
        boundaries = np.concatenate(([0], change_idx, [len(atividades)]))

        for b in range(len(boundaries) - 1):
            i0, i1 = boundaries[b], boundaries[b + 1]
            bloco = dados_dev[i0:i1]
            if bloco.shape[0] < win_size:
                continue
            atividade = int(bloco[0, COL_ACTIVITY])

            # construir matriz 9-colunas
            X9 = np.stack([
                bloco[:, COL_ACC_X], bloco[:, COL_ACC_Y], bloco[:, COL_ACC_Z],
                bloco[:, COL_GYRO_X], bloco[:, COL_GYRO_Y], bloco[:, COL_GYRO_Z],
                bloco[:, COL_MAG_X], bloco[:, COL_MAG_Y], bloco[:, COL_MAG_Z],
            ], axis=1)

            for s, e in _windows_indices_4_2(bloco.shape[0], win_size, hop_size):
                seg = X9[s:e]
                if seg.shape[0] != win_size:
                    continue
                segmentos.append({
                    'device': sensor_id,
                    'activity': atividade,
                    'X': seg
                })

    return segmentos


def segmentar_janelas_com_sujeito(dados_raw, fs=51.2, janela_s=5.0, overlap=0.5):
    """
    (Tarefa 3 Meta 2) Segmenta janelas mantendo a informação do 'subject'.
    Processa cada sujeito individualmente para evitar misturar dados.
    Assume que dados_raw tem a coluna COL_PARTICIPANT (coluna 12).
    """
    win_size = int(janela_s * fs)
    hop_size = int(win_size * (1.0 - overlap))
    segmentos = []

    # Identificar sujeitos únicos na coluna 12
    sujeitos_unicos = np.unique(dados_raw[:, COL_PARTICIPANT])

    for subj_id in sujeitos_unicos:
        # Filtrar dados deste sujeito
        mask_subj = dados_raw[:, COL_PARTICIPANT] == subj_id
        dados_subj = dados_raw[mask_subj]

        for sensor_id in range(1, 6):
            mask_dev = dados_subj[:, COL_DEVICE_ID] == sensor_id
            dados_dev = dados_subj[mask_dev]
            
            if dados_dev.size == 0:
                continue

            # Ordenar por tempo
            idx_sort = np.argsort(dados_dev[:, COL_TIMESTAMP])
            dados_dev = dados_dev[idx_sort]

            # Blocos de atividade contínua
            atividades = dados_dev[:, COL_ACTIVITY].astype(int)
            change_idx = np.where(np.diff(atividades) != 0)[0] + 1
            boundaries = np.concatenate(([0], change_idx, [len(atividades)]))

            for b in range(len(boundaries) - 1):
                i0, i1 = boundaries[b], boundaries[b + 1]
                bloco = dados_dev[i0:i1]
                if bloco.shape[0] < win_size:
                    continue
                
                atividade = int(bloco[0, COL_ACTIVITY])
                
                # Matriz 9 colunas (Acc, Gyro, Mag)
                X9 = np.stack([
                    bloco[:, COL_ACC_X], bloco[:, COL_ACC_Y], bloco[:, COL_ACC_Z],
                    bloco[:, COL_GYRO_X], bloco[:, COL_GYRO_Y], bloco[:, COL_GYRO_Z],
                    bloco[:, COL_MAG_X], bloco[:, COL_MAG_Y], bloco[:, COL_MAG_Z],
                ], axis=1)

                for s, e in _windows_indices_4_2(bloco.shape[0], win_size, hop_size):
                    seg = X9[s:e]
                    if seg.shape[0] == win_size:
                        segmentos.append({
                            'device': sensor_id,
                            'activity': atividade,
                            'subject': int(subj_id), # <--- Guardamos o sujeito aqui
                            'X': seg
                        })
    
    return segmentos


def _feature_vector_series_temporal_freq_4_2(series, fs):
    feats = []
    # Temporais
    mean = np.mean(series)
    std = np.std(series)
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    rms = np.sqrt(np.mean(series ** 2))
    wl = np.sum(np.abs(np.diff(series)))  # waveform length
    # zero-crossing rate
    zc = np.sum(series[:-1] * series[1:] < 0) / (len(series) - 1)
    feats += [mean, std, median, mad, rms, wl, zc]

    # Espectrais (periodograma)
    f, Pxx = signal.periodogram(series, fs=fs, scaling='spectrum', window='hann')
    Pxx = np.maximum(Pxx, 1e-12)
    energy = np.sum(Pxx)
    Pnorm = Pxx / np.sum(Pxx)
    spec_entropy = -np.sum(Pnorm * np.log(Pnorm))
    dom_idx = np.argmax(Pxx)
    dom_freq = f[dom_idx]
    dom_amp = Pxx[dom_idx]
    # centróide espectral
    spec_centroid = np.sum(f * Pxx) / np.sum(Pxx)
    feats += [energy, spec_entropy, dom_freq, dom_amp, spec_centroid]
    return feats


def extrair_features_janela_4_2(seg, fs=51.2):
    """
    Recebe um segmento com matriz [N,9] (acc, gyro, mag) e extrai features
    temporais e espectrais para 12 séries: 9 eixos + 3 módulos.
    Retorna (feature_vector, feature_names)
    """
    X = seg['X']
    # módulos por amostra
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
            f"{base}_{name}_mean",
            f"{base}_{name}_std",
            f"{base}_{name}_median",
            f"{base}_{name}_mad",
            f"{base}_{name}_rms",
            f"{base}_{name}_wl",
            f"{base}_{name}_zcr",
            f"{base}_{name}_spec_energy",
            f"{base}_{name}_spec_entropy",
            f"{base}_{name}_spec_dom_freq",
            f"{base}_{name}_spec_dom_amp",
            f"{base}_{name}_spec_centroid",
        ])

    return np.array(feat_vec, dtype=float), feat_names


def construir_feature_set_4_2(dados_raw, fs=51.2):
    """
    Constrói matriz de features e rótulos por janela pura (5s, 50% overlap) por dispositivo.
    Retorna X, y, feature_names
    """
    segmentos = segmentar_janelas_puras_4_2(dados_raw, fs=fs, janela_s=5.0, overlap=0.5)
    X_list, y_list = [], []
    feat_names_ref = None
    for seg in segmentos:
        fv, fn = extrair_features_janela_4_2(seg, fs)
        if feat_names_ref is None:
            feat_names_ref = fn
        X_list.append(fv)
        y_list.append(seg['activity'])
    if not X_list:
        return np.zeros((0, 0)), np.array([]), []
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, feat_names_ref


# 4.3/4.4 PCA e variância explicada; normalização z-score
def pca_pipeline_4_3_4_4(X, variancia_target=0.75):
    if X.shape[0] == 0:
        return None, None, None, None
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
    """
    Guarda CSVs úteis para avaliação: feature matrix, labels, nomes, variância PCA.
    """
    try:
        np.savetxt("tarefa4_features_X.csv", X, delimiter=",")
        np.savetxt("tarefa4_labels_y.csv", y, fmt="%d", delimiter=",")
        with open("tarefa4_feature_names.csv", "w") as f:
            f.write("feature\n")
            for n in feature_names:
                f.write(f"{n}\n")
        if pca_explained is not None:
            np.savetxt("tarefa4_pca_explained_variance_ratio.csv", pca_explained, delimiter=",")
        if cvar is not None:
            np.savetxt("tarefa4_pca_cumulative_variance.csv", cvar, delimiter=",")
            # plot cumulativa
            plt.figure(figsize=(6,4))
            plt.plot(np.arange(1, len(cvar)+1), cvar, marker='o')
            plt.axhline(0.75, color='red', linestyle='--', label='75%')
            plt.xlabel('Número de Componentes')
            plt.ylabel('Variância explicada cumulativa')
            plt.title('PCA: Variância explicada cumulativa')
            plt.legend()
            plt.tight_layout()
            plt.savefig("graphs/tarefa4_pca_variancia.png")
    except Exception as e:
        print(f"[Aviso] Falha ao guardar artefactos de Tarefa 4: {e}")


def guardar_rankings_4_6(scores, feature_names, fname):
    try:
        order = np.argsort(np.abs(scores))[::-1]
        with open(fname, "w") as f:
            f.write("rank,feature,score,abs_score\n")
            for r, j in enumerate(order, 1):
                f.write(f"{r},{feature_names[j]},{scores[j]},{abs(scores[j])}\n")
    except Exception as e:
        print(f"[Aviso] Falha ao guardar ranking em {fname}: {e}")


def selecionar_features_topk_4_6(X, feature_names, scores, topk=10):
    """
    Seleciona as top-k features por |score| e devolve X_sel, names_sel, idx_sel.
    """
    order = np.argsort(np.abs(scores))[::-1][:topk]
    X_sel = X[:, order]
    names_sel = [feature_names[j] for j in order]
    return X_sel, names_sel, order


# 4.5 Fisher Score e ReliefF
def fisher_score_multi_4_5(X, y):
    """
    Fisher Score multi-classe por feature.
    score_j = sum_c n_c (mu_c - mu)^2 / sum_c n_c sigma_c^2
    """
    n_classes = np.unique(y)
    mu = X.mean(axis=0)
    scores = np.zeros(X.shape[1], dtype=float)
    for c in n_classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        if nc == 0:
            continue
        mu_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0) + 1e-12
        scores += nc * (mu_c - mu) ** 2 / var_c
    return scores


def relieff_4_5(X, y, n_neighbors=10, n_samples=2000, random_state=0):
    """
    Implementação simples do ReliefF multi-classe.
    """
    rng = np.random.default_rng(random_state)
    m = X.shape[0]
    if m == 0:
        return np.zeros(X.shape[1])
    idx_samples = rng.choice(m, size=min(n_samples, m), replace=False)
    W = np.zeros(X.shape[1], dtype=float)
    # pré-computar por classe
    classes = np.unique(y)
    from sklearn.neighbors import NearestNeighbors
    nbrs_all = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    for i in idx_samples:
        x_i = X[i : i + 1]
        yi = y[i]
        # vizinhos do mesmo e de outras classes
        distances, indices = nbrs_all.kneighbors(x_i)
        indices = indices.flatten()[1:]
        same = indices[y[indices] == yi][:n_neighbors]
        # difer classe: escolher k mais próximos por classe e fazer média
        for c in classes:
            if c == yi:
                continue
            idx_c = indices[y[indices] == c][:n_neighbors]
            if idx_c.size == 0:
                continue
            diff = np.abs(X[i] - X[idx_c]).mean(axis=0)
            W += diff / (len(classes) - 1)
        if same.size > 0:
            diff_hit = np.abs(X[i] - X[same]).mean(axis=0)
            W -= diff_hit
    # normalizar por número de amostras e vizinhos
    W /= max(1, len(idx_samples))
    return W


def reportar_top_features_4_6(scores, feature_names, topk=10, titulo=""):
    magnitudes = np.abs(scores)
    order = np.argsort(magnitudes)[::-1][:topk]
    print(f"\nTop-{topk} features {titulo}:")
    for rank, j in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_names[j]}  (score={scores[j]:.4f}, |score|={magnitudes[j]:.4f})")




# ///// META 2 /////

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
    
    # y já deve estar filtrado, mas garantir
    mask = y <= ATIVIDADE_META2_MAX
    y_subset = y[mask]
    
    classes, counts = np.unique(y_subset, return_counts=True)
    
    print("Contagem de amostras por atividade:")
    total = 0
    for cls, count in zip(classes, counts):
        print(f"  Atividade {int(cls)}: {count} amostras")
        total += count
    
    print(f"Total de amostras (atividades 1-7): {total}")
        
    # Verificar balanceamento (ex: desvio padrão das contagens)
    std_counts = np.std(counts)
    media_counts = np.mean(counts)
    cv = std_counts / media_counts # Coeficiente de variação
    
    print(f"\nEstatísticas de balanceamento:")
    print(f"  Média: {media_counts:.1f}")
    print(f"  Desvio padrão: {std_counts:.1f}")
    print(f"  Coeficiente de variação: {cv:.3f}")
    
    if cv < 0.2: # Limiar arbitrário para "balanceado"
        print("-> O dataset parece razoavelmente balanceado.")
    else:
        print("-> O dataset NÃO está balanceado (diferenças significativas nas contagens).")
    
    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribuição das Atividades (1-7)", fontsize=16, fontweight='bold')
    plt.xlabel("Atividade", fontsize=14)
    plt.ylabel("Número de Segmentos", fontsize=14)
    plt.xticks(classes)
    plt.grid(axis='y', alpha=0.5)
    
    # Adicionar valores no topo das barras
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
    # Configurar random state para reprodutibilidade
    np.random.seed(random_state)
    
    # 1. Selecionar todas as amostras da classe alvo
    mask = y == atividade_alvo
    X_classe = X[mask]
    n_amostras = X_classe.shape[0]
    
    if n_amostras < 2:
        print(f"[Aviso] Amostras insuficientes ({n_amostras}) para SMOTE na atividade {atividade_alvo}.")
        return np.zeros((0, X.shape[1]))

    # Ajustar k_neighbors se houver poucas amostras
    k_neighbors = min(k_neighbors, n_amostras - 1)
    if k_neighbors < 1:
        print(f"[Aviso] k_neighbors ajustado para {k_neighbors}, mas é insuficiente.")
        return np.zeros((0, X.shape[1]))

    # 2. Fit k-Nearest Neighbors
    # Usamos k+1 porque o primeiro vizinho é o próprio ponto
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(X_classe)
    
    amostras_sinteticas = []
    
    print(f"  A gerar {K} amostras sintéticas para atividade {atividade_alvo}...")
    
    for i in range(K):
        # a. Escolher aleatoriamente uma amostra base (índice i)
        idx_base = np.random.randint(0, n_amostras)
        vetor_base = X_classe[idx_base]
        
        # b. Encontrar os k vizinhos mais próximos
        distancias, indices = nbrs.kneighbors(vetor_base.reshape(1, -1))
        # O indices[0] contém os índices em X_classe. O primeiro é ele próprio.
        vizinhos_indices = indices[0][1:]
        
        # c. Escolher aleatoriamente um dos vizinhos
        idx_vizinho = np.random.choice(vizinhos_indices)
        vetor_vizinho = X_classe[idx_vizinho]
        
        # d. Interpolar: novo = base + rand * (vizinho - base)
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
    
    # 1. Carregar dados APENAS do participante 3
    print("A carregar dados do Participante 3...")
    dados_p3 = carregar_dados_participante(3, base_dir=base_dir)
    
    if dados_p3.size == 0:
        print("[Erro] Não foi possível carregar dados do participante 3.")
        return

    # 2. Extrair features para este participante
    print("A extrair features do participante 3...")
    X_p3, y_p3, feat_names = construir_feature_set_4_2(dados_p3, fs=51.2)
    
    if X_p3.shape[0] == 0:
        print("[Erro] Nenhuma feature extraída para o participante 3.")
        return
    
    print(f"Features extraídas: {X_p3.shape}")
    
    # Filtrar apenas atividades 1-7 (conforme constante ATIVIDADE_META2_MAX)
    mask_valid = y_p3 <= ATIVIDADE_META2_MAX
    X_p3 = X_p3[mask_valid]
    y_p3 = y_p3[mask_valid]
    
    print(f"Após filtro (ativ 1-{ATIVIDADE_META2_MAX}): {X_p3.shape}")

    # 3. Gerar 3 amostras sintéticas para atividade 4
    atividade_alvo = 4
    K = 3
    print(f"\nA gerar {K} amostras sintéticas para a atividade {atividade_alvo}...")
    X_sintetico = gerar_smote_1_2(X_p3, y_p3, atividade_alvo, K=K, k_neighbors=5)
    
    if X_sintetico.shape[0] != K:
        print("[Erro] Falha ao gerar amostras sintéticas.")
        return
    
    print(f"Amostras sintéticas geradas com sucesso: {X_sintetico.shape}")
    
    # 4. Visualização 2D (primeiras 2 features)
    # Criar figura com 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Subplot 1: Features 0 vs 1 ---
    ax1 = axes[0]
    
    # Plotar amostras originais (coloridas por atividade)
    classes_presentes = np.unique(y_p3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes_presentes)))
    
    for cls, color in zip(classes_presentes, colors):
        mask = y_p3 == cls
        label_text = f'Atividade {int(cls)}'
        if cls == atividade_alvo:
            label_text += ' (Original)'
        ax1.scatter(X_p3[mask, 0], X_p3[mask, 1], 
                   alpha=0.5, s=30, color=color, label=label_text)
    
    # Plotar amostras sintéticas (destacadas)
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
    
    # --- Subplot 2: Features 2 vs 3 (visualização alternativa) ---
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
    
    # 5. Imprimir estatísticas das amostras geradas
    print("\n--- Estatísticas das Amostras Sintéticas ---")
    print(f"Forma: {X_sintetico.shape}")
    print(f"Primeiras 5 features da 1ª amostra sintética:")
    print(X_sintetico[0, :5])
    
    # Comparar com média das amostras originais da atividade 4
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

    # 1. Preparar os dados (Reamostragem)
    resampled_segments = []
    
    for i, seg in enumerate(segmentos):
        # O modelo só usa Acelerómetro (colunas 0, 1, 2 do teu 'X')
        acc_data = seg['X'][:, 0:3]
        
        # Aplicar a função do ficheiro fornecido
        # Nota: a função retorna (dados, fs), por isso usamos [0]
        acc_30hz, _ = resample_to_30hz_5s(acc_data, fs_in_hz=fs_original)
        
        resampled_segments.append(acc_30hz)

    # Converter para numpy array: Shape (N_segmentos, 150, 3)
    # 150 amostras = 5 segundos * 30Hz
    x_all = np.array(resampled_segments)
    
    # 2. Transpor para o formato esperado pelo PyTorch (Batch, Canais, Tempo)
    # De (N, 150, 3) para (N, 3, 150)
    x_all = np.transpose(x_all, (0, 2, 1))
    
    print(f"  Shape de entrada no modelo: {x_all.shape}")

    # 3. Carregar o Modelo
    try:
        feature_encoder = load_model()
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o modelo (verifique a internet/PyTorch): {e}")
        return np.array([])

    # 4. Inferência em Batches (para gerir memória)
    embeddings_list = []
    batch_size = 32 # Processa 32 segmentos de cada vez
    
    with torch.no_grad(): # Desativa gradientes para poupar memória e ser mais rápido
        for i in range(0, x_all.shape[0], batch_size):
            # Selecionar batch
            batch_numpy = x_all[i : i + batch_size]
            
            # Converter para Tensor PyTorch (Float)
            batch_tensor = torch.from_numpy(batch_numpy).float().to("cpu")
            
            # Passar pelo modelo
            emb_batch = feature_encoder(batch_tensor)
            
            # Guardar resultado (converter de volta para numpy)
            embeddings_list.append(emb_batch.cpu().numpy())
            
            if (i // batch_size) % 5 == 0:
                print(f"    Processado batch {i // batch_size}...")

    # Concatenar todos os resultados
    embeddings_final = np.concatenate(embeddings_list, axis=0)
    print(f"  Embeddings gerados com sucesso. Shape final: {embeddings_final.shape}")
    
    return embeddings_final


# --- TAREFA 3: Validação e Classificação ---

def split_within_subject_3_1(X, y, subjects, X_emb=None, train_r=0.6, val_r=0.2, random_state=42):
    """
    (Tarefa 3.1) Divide os dados em Train/Val/Test mantendo as proporções 
    DENTRO de cada sujeito.
    
    Parâmetros:
        X: matriz de features (n_amostras, n_features)
        y: vetor de labels (n_amostras,)
        subjects: vetor de IDs de sujeitos (n_amostras,)
        X_emb: matriz de embeddings opcional (n_amostras, n_emb_features)
        train_r: proporção de treino (default: 0.6 = 60%)
        val_r: proporção de validação (default: 0.2 = 20%)
        random_state: seed para reprodutibilidade
    
    Retorna:
        dict com conjuntos train/val/test para features, embeddings (se fornecidos), labels e subjects
    """
    print(f"\n--- Tarefa 3.1: Split Within-Subject ({train_r*100:.0f}-{val_r*100:.0f}-{(1-train_r-val_r)*100:.0f}%) ---")
    
    np.random.seed(random_state)
    
    indices_train = []
    indices_val = []
    indices_test = []
    
    # Lista de sujeitos únicos
    unique_subs = np.unique(subjects)
    print(f"Número de sujeitos únicos: {len(unique_subs)}")
    
    for s in unique_subs:
        # Índices onde o sujeito é 's'
        idx_s = np.where(subjects == s)[0]
        
        # Baralhar índices deste sujeito
        np.random.shuffle(idx_s)
        
        n = len(idx_s)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        
        # Fatiar
        train_i = idx_s[:n_train]
        val_i = idx_s[n_train : n_train + n_val]
        test_i = idx_s[n_train + n_val:]
        
        indices_train.extend(train_i)
        indices_val.extend(val_i)
        indices_test.extend(test_i)
    
    # Converter para arrays numpy para indexação
    idx_train = np.array(indices_train)
    idx_val = np.array(indices_val)
    idx_test = np.array(indices_test)
    
    # Construir dicionário de retorno
    data_split = {
        'X_train': X[idx_train], 'y_train': y[idx_train],
        'X_val':   X[idx_val],   'y_val':   y[idx_val],
        'X_test':  X[idx_test],  'y_test':  y[idx_test],
        'sub_train': subjects[idx_train],
        'sub_val':   subjects[idx_val],
        'sub_test':  subjects[idx_test]
    }
    
    # Adicionar embeddings se fornecidos
    if X_emb is not None:
        data_split['X_emb_train'] = X_emb[idx_train]
        data_split['X_emb_val']   = X_emb[idx_val]
        data_split['X_emb_test']  = X_emb[idx_test]
        print(f"  Embeddings também divididos.")
    
    print(f"  Train: {len(idx_train)} amostras | Val: {len(idx_val)} amostras | Test: {len(idx_test)} amostras")
    print(f"  Distribuição Train: {dict(zip(*np.unique(y[idx_train], return_counts=True)))}")
    print(f"  Distribuição Val: {dict(zip(*np.unique(y[idx_val], return_counts=True)))}")
    print(f"  Distribuição Test: {dict(zip(*np.unique(y[idx_test], return_counts=True)))}")
    
    return data_split


def split_between_subject_3_2(X, y, subjects, X_emb=None, train_subs=9, val_subs=3, test_subs=3, random_state=42):
    """
    (Tarefa 3.2) Divide os dados em Train/Val/Test ao nível do SUJEITO.
    O modelo treina com uns participantes e testa noutros totalmente novos.
    """
    print(f"\n--- Tarefa 3.2: Split Between-Subject ({train_subs} treinar, {val_subs} validar, {test_subs} testar) ---")
    
    np.random.seed(random_state)
    
    # 1. Obter lista de sujeitos únicos disponíveis nos dados
    unique_subs = np.unique(subjects)
    n_total = len(unique_subs)
    
    if n_total < (train_subs + val_subs + test_subs):
        print(f"[Aviso] Número de sujeitos ({n_total}) inferior ao pedido. Ajustando...")
        # Lógica de fallback simples (ex: resto vai para teste)
    
    # 2. Baralhar os SUJEITOS (não as linhas)
    np.random.shuffle(unique_subs)
    
    # 3. Definir quais sujeitos vão para onde
    subs_train = unique_subs[:train_subs]
    subs_val   = unique_subs[train_subs : train_subs + val_subs]
    subs_test  = unique_subs[train_subs + val_subs :]
    
    print(f"  Sujeitos Treino: {subs_train}")
    print(f"  Sujeitos Validação: {subs_val}")
    print(f"  Sujeitos Teste: {subs_test}")
    
    # 4. Criar máscaras booleanas baseadas nos IDs
    mask_train = np.isin(subjects, subs_train)
    mask_val   = np.isin(subjects, subs_val)
    mask_test  = np.isin(subjects, subs_test)
    
    # 5. Construir dicionário de retorno
    data_split = {
        'X_train': X[mask_train], 'y_train': y[mask_train],
        'X_val':   X[mask_val],   'y_val':   y[mask_val],
        'X_test':  X[mask_test],  'y_test':  y[mask_test],
        'sub_train': subjects[mask_train], # Útil para debug
        'sub_val':   subjects[mask_val],
        'sub_test':  subjects[mask_test]
    }
    
    if X_emb is not None:
        data_split['X_emb_train'] = X_emb[mask_train]
        data_split['X_emb_val']   = X_emb[mask_val]
        data_split['X_emb_test']  = X_emb[mask_test]
        
    print(f"  Train: {data_split['y_train'].shape[0]} amostras")
    print(f"  Val:   {data_split['y_val'].shape[0]} amostras")
    print(f"  Test:  {data_split['y_test'].shape[0]} amostras")
    
    return data_split


def processar_cenarios_3_4(X_train, X_val, X_test, y_train, top_k=15, pca_var=0.90):
    """
    (Tarefa 3.4) Gera os 3 cenários (All, PCA, ReliefF) garantindo que
    o 'fit' é feito apenas no conjunto de TREINO.
    """
    cenarios = {}

    # 1. Normalização (Obrigatória para kNN e PCA)
    # Fit no Train -> Transform no Train, Val, Test
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm   = scaler.transform(X_val)
    X_test_norm  = scaler.transform(X_test)

    # --- CENÁRIO A: Todas as Features (Normalizadas) ---
    cenarios['all'] = {
        'X_train': X_train_norm, 'X_val': X_val_norm, 'X_test': X_test_norm
    }

    # --- CENÁRIO B: PCA (90% Variância) ---
    print(f"  > A ajustar PCA (var={pca_var*100:.0f}%)...")
    pca = PCA(n_components=pca_var)
    # Fit apenas no Train
    pca.fit(X_train_norm)
    
    cenarios['pca'] = {
        'X_train': pca.transform(X_train_norm),
        'X_val':   pca.transform(X_val_norm),
        'X_test':  pca.transform(X_test_norm)
    }
    print(f"    Componentes mantidas: {pca.n_components_}")

    # --- CENÁRIO C: ReliefF (Top 15) ---
    print(f"  > A executar ReliefF (Top {top_k})...")
    # Usa a tua função relieff_4_5 existente.
    # Nota: O y_train é necessário para calcular os scores.
    # O n_samples limita o cálculo para ser rápido (o enunciado não proíbe subsampling para o ranking)
    scores = relieff_4_5(X_train_norm, y_train, n_neighbors=10, n_samples=2000)
    
    # Selecionar índices das melhores features
    order = np.argsort(np.abs(scores))[::-1] # Ordem decrescente
    top_indices = order[:top_k]
    
    cenarios['relief'] = {
        'X_train': X_train_norm[:, top_indices],
        'X_val':   X_val_norm[:, top_indices],
        'X_test':  X_test_norm[:, top_indices]
    }
    
    return cenarios

def pipeline_preparacao_3_4(split_data):
    """
    Orquestra o processamento para Features Clássicas e Embeddings.
    Recebe o dicionário 'split_data' da Tarefa 3.1 ou 3.2.
    """
    resultados = {}
    
    # 1. Processar FEATURES CLÁSSICAS
    print("\n--- Tarefa 3.4: Preparar Cenários (FEATURES) ---")
    resultados['features'] = processar_cenarios_3_4(
        split_data['X_train'], split_data['X_val'], split_data['X_test'],
        split_data['y_train']
    )
    
    # 2. Processar EMBEDDINGS (se existirem)
    if 'X_emb_train' in split_data:
        print("\n--- Tarefa 3.4: Preparar Cenários (EMBEDDINGS) ---")
        X_train_e = split_data['X_emb_train']
        X_val_e   = split_data['X_emb_val']
        X_test_e  = split_data['X_emb_test']
        
        # Garantir que é 2D: (N, 512) e não (N, 512, 1)
        if X_train_e.ndim > 2:
            X_train_e = X_train_e.reshape(X_train_e.shape[0], -1)
            X_val_e   = X_val_e.reshape(X_val_e.shape[0], -1)
            X_test_e  = X_test_e.reshape(X_test_e.shape[0], -1)

        resultados['embeddings'] = processar_cenarios_3_4(
            X_train_e, X_val_e, X_test_e,
            split_data['y_train']
        )
        
    return resultados



# --- Função Principal (main)  ---
def main():
    """
    Função principal que orquestra a execução das tarefas da Meta 1 e Meta 2.
    """

    # --- PAINEL DE CONTROLO ---
    # Defina como True/False para executar/saltar cada tarefa

    # META 1: Tarefa 2: Carregar dados de 1 participante (teste rápido)
    RUN_TASK_2_TEST = False

    # META 1: Tarefa 3.1: Gerar os 15 boxplots (Requer 'dados_todos')
    RUN_TASK_3_1 = False

    # META 1: Tarefa 3.2: Calcular densidade IQR (Pulso Direito)
    RUN_TASK_3_2 = False

    # META 1: Tarefa 3.4: Gerar plots Z-Score (Demorado: 3x15 plots)
    RUN_TASK_3_4 = False

    # META 1: Tarefa 3.5: Gerar tabela comparativa de densidades
    RUN_TASK_3_5 = False

    # META 1: Tarefa 3.7: Gerar plots K-Means 3D (Exemplo focado)
    RUN_TASK_3_7 = False

    # META 1: Tarefa 3.7.1 (Bónus): Gerar plot DBSCAN 3D
    RUN_TASK_3_7_1 = False

    # META 1: Tarefa 4: Extração de informação característica
    RUN_TASK_4_1 = False
    RUN_TASK_4_2 = False  # janelação + features
    RUN_TASK_4_3_4_4 = False  # PCA e variância
    RUN_TASK_4_5_4_6 = False  # Fisher, ReliefF e top-10

    # META 2: Tarefa 1 - Data Augmentation
    RUN_META2_TASK_1_1 = True  # Análise de balanceamento
    RUN_META2_TASK_1_3 = True  # Visualização SMOTE
    
    # META 2: Tarefa 2 - Embeddings
    RUN_META2_TASK_2_1 = True  # Gerar Embeddings Dataset
    
    # META 2: Tarefa 3 - Validação e Classificação
    RUN_META2_TASK_3_1 = True  # Split Within-Subject
    RUN_META2_TASK_3_2 = True  # Split Between-Subject
    RUN_META2_TASK_3_4 = True  # Preparação de cenários (PCA, ReliefF, etc.)

    # --- FIM PAINEL DE CONTROLO ---

    # --- Execução ---

    if RUN_TASK_2_TEST:
        print("--- A executar Tarefa 2 (Teste) ---")
        # Testar a Tarefa 2
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dados_p1 = carregar_dados_participante(1, base_dir=script_dir)
        print(f"Dados do participante 1 carregados. Dimensão: {dados_p1.shape}")
        if dados_p1.size == 0:
            print("[ERRO] Teste da Tarefa 2 falhou. Verifique o caminho.")
            return # Sai se não conseguir carregar dados de teste

    # Para as tarefas 3.1 em diante, precisamos de *todos* os dados
    # Usamos estas flags para só carregar os dados se for necessário
    if any([RUN_TASK_3_1, RUN_TASK_3_2, RUN_TASK_3_4, RUN_TASK_3_5, RUN_TASK_3_7, RUN_TASK_3_7_1, RUN_TASK_4_1, RUN_TASK_4_2, RUN_TASK_4_3_4_4, RUN_TASK_4_5_4_6]):

        # --- Passo 1: Carregar Dados (Raw) ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dados_todos = carregar_dados_todos_participantes(base_dir=script_dir)
        if dados_todos.size == 0:
            return # Sai se não houver dados

        # --- Passo 2: Transformar Dados (Módulos)  ---
        # (Apenas necessário para 3.1, 3.2, 3.4, 3.5)
        if any([RUN_TASK_3_1, RUN_TASK_3_2, RUN_TASK_3_4, RUN_TASK_3_5, RUN_TASK_4_1]):
            dados_transformados = get_dados_transformados(dados_todos)

            if RUN_TASK_3_1:
                plotar_boxplots_3_1(dados_transformados)

            if RUN_TASK_3_2:
                analisar_densidade_iqr_3_2(dados_transformados)

            if RUN_TASK_3_4:
                plotar_outliers_zscore_3_4(dados_transformados)

            if RUN_TASK_3_5:
                comparar_densidades_3_5(dados_transformados)

            if RUN_TASK_4_1:
                testar_significancia_medias_4_1(dados_transformados)

        # --- Passo 3: Análise Multivariada (Usa dados 'raw') ---
        # (Tarefas 3.7 e 3.7.1 usam os dados 'raw' (x,y,z))

        if RUN_TASK_3_7:
            # Testa com 2, 3 e 5 clusters
            analisar_outliers_kmeans_3_7(dados_todos, n_clusters_lista=[2, 3, 5])

        if RUN_TASK_3_7_1:
            analisar_outliers_dbscan_3_7_1(dados_todos)

        # --- Tarefas 4.2 em diante usam dados raw para janelação/feature set ---
        if RUN_TASK_4_2 or RUN_TASK_4_3_4_4 or RUN_TASK_4_5_4_6:
            print("\nA construir feature set (Tarefa 4.2)...")
            X, y, feat_names = construir_feature_set_4_2(dados_todos, fs=51.2)
            print(f"Feature set: X={X.shape}, y={y.shape}")

            if RUN_TASK_4_3_4_4:
                print("\nA executar PCA (Tarefas 4.3/4.4)...")
                scaler, pca_n, Xp, cvar = pca_pipeline_4_3_4_4(X, variancia_target=0.75)
                if Xp is not None:
                    n_comp = Xp.shape[1]
                    print(f"Número de componentes para >=75% variância: {n_comp}")
                    # 4.4.1 Exemplo para um instante
                    exemplo_idx = 0
                    Xz = scaler.transform(X[exemplo_idx:exemplo_idx+1])
                    exemplo_pca = pca_n.transform(Xz)[0]
                    print(f"Exemplo (instante {exemplo_idx}) features PCA: {exemplo_pca[:min(10, len(exemplo_pca))]}")

                    # Guardar artefactos PCA e feature set
                    try:
                        full_pca = PCA()
                        Xz_all = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
                        full_pca.fit(Xz_all)
                        guardar_artifacts_4_x(X, y, feat_names, cvar=np.cumsum(full_pca.explained_variance_ratio_), pca_explained=full_pca.explained_variance_ratio_)
                    except Exception:
                        guardar_artifacts_4_x(X, y, feat_names, cvar=cvar)

            if RUN_TASK_4_5_4_6 and X.shape[0] > 0:
                print("\nA calcular Fisher Score (4.5) e ReliefF (4.5) e Top-10 (4.6)...")
                # Normalizar para ReliefF (melhor comportamento)
                Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
                fisher = fisher_score_multi_4_5(Xn, y)
                rel = relieff_4_5(Xn, y, n_neighbors=10, n_samples=min(2000, Xn.shape[0]))
                reportar_top_features_4_6(fisher, feat_names, topk=10, titulo="(Fisher)")
                reportar_top_features_4_6(rel, feat_names, topk=10, titulo="(ReliefF)")
                # Guardar rankings completos
                guardar_rankings_4_6(fisher, feat_names, "tarefa4_fisher_ranking.csv")
                guardar_rankings_4_6(rel, feat_names, "tarefa4_relieff_ranking.csv")

                # 4.6.1 Obter features relativas à seleção e exemplificar num instante
                topk = 10
                Xf, names_f, idx_f = selecionar_features_topk_4_6(Xn, feat_names, fisher, topk=topk)
                Xr, names_r, idx_r = selecionar_features_topk_4_6(Xn, feat_names, rel, topk=topk)
                exemplo_idx = 0
                print(f"\n[4.6.1] Exemplo (instante {exemplo_idx}) - Vetor selecionado Fisher (k={topk}):")
                print(Xf[exemplo_idx])
                print(f"[4.6.1] Exemplo (instante {exemplo_idx}) - Vetor selecionado ReliefF (k={topk}):")
                print(Xr[exemplo_idx])
                try:
                    np.savetxt("tarefa4_selected_fisher_X.csv", Xf, delimiter=",")
                    with open("tarefa4_selected_fisher_feature_names.csv", "w") as f:
                        f.write("feature\n")
                        for n in names_f:
                            f.write(f"{n}\n")
                    np.savetxt("tarefa4_selected_relieff_X.csv", Xr, delimiter=",")
                    with open("tarefa4_selected_relieff_feature_names.csv", "w") as f:
                        f.write("feature\n")
                        for n in names_r:
                            f.write(f"{n}\n")
                except Exception as e:
                    print(f"[Aviso] Falha ao guardar features selecionadas: {e}")

    # ==========================================
    # --- META 2 (MÓDULO B) ---
    # ==========================================
    
    if RUN_META2_TASK_1_1 or RUN_META2_TASK_1_3 or RUN_META2_TASK_2_1 or RUN_META2_TASK_3_1:
        print("\n" + "="*60)
        print("=== META 2 (MÓDULO B): DATA AUGMENTATION & EMBEDDINGS ===")
        print("="*60)
        print(f"Nota: Módulo B considera APENAS atividades 1-{ATIVIDADE_META2_MAX}\n")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. CARREGAR DADOS COM IDs DE PARTICIPANTES
        print("--- Etapa 1: Carregamento de Dados com IDs ---")
        
        # Verificar se dados_todos já tem a coluna de IDs (coluna 12)
        if 'dados_todos' not in locals() or dados_todos.shape[1] < 13:
            print("A carregar dados COM IDs de participantes...")
            dados_todos = carregar_dados_todos_com_id(base_dir=script_dir)
        else:
            print("Dados já carregados com IDs.")
        
        if dados_todos.size == 0:
            print("[ERRO] Nenhum dado disponível. A saltar Meta 2.")
        else:
            # 2. SEGMENTAÇÃO COM SUJEITOS
            print("\n--- Etapa 2: Segmentação de Dados (5s, 50% overlap, COM subject) ---")
            
            # Verificar se segmentos já têm informação de 'subject'
            if 'segmentos' not in locals() or (len(segmentos) > 0 and 'subject' not in segmentos[0]):
                print("A segmentar janelas COM informação de sujeitos...")
                segmentos = segmentar_janelas_com_sujeito(dados_todos, fs=51.2, janela_s=5.0, overlap=0.5)
            else:
                print("Segmentos já existem com informação de sujeitos.")
            
            # Extrair vetores auxiliares alinhados
            y_todos = np.array([seg['activity'] for seg in segmentos], dtype=int)
            subjects_todos = np.array([seg['subject'] for seg in segmentos], dtype=int)
            
            print(f"Total de segmentos gerados: {len(segmentos)}")
            print(f"Distribuição de atividades: {dict(zip(*np.unique(y_todos, return_counts=True)))}")
            print(f"Número de sujeitos únicos: {len(np.unique(subjects_todos))}")
            
            # 3. FILTRAGEM GLOBAL (Apenas Atividades 1-7 do Módulo B)
            print(f"\n--- Etapa 3: Filtragem para Atividades 1-{ATIVIDADE_META2_MAX} ---")
            mask_mod_b = y_todos <= ATIVIDADE_META2_MAX
            
            segmentos_mod_b = [seg for i, seg in enumerate(segmentos) if mask_mod_b[i]]
            y_mod_b = y_todos[mask_mod_b]
            subjects_mod_b = subjects_todos[mask_mod_b]
            
            print(f"Segmentos após filtro: {len(segmentos_mod_b)}")
            print(f"Distribuição filtrada: {dict(zip(*np.unique(y_mod_b, return_counts=True)))}")
            print(f"Sujeitos presentes: {np.unique(subjects_mod_b)}")
            
            # 4. TAREFA 1: Data Augmentation (Features Manuais)
            if RUN_META2_TASK_1_1 or RUN_META2_TASK_1_3:
                print("\n" + "-"*60)
                print("--- TAREFA 1: Data Augmentation (Features Manuais) ---")
                print("-"*60)
                
                if RUN_META2_TASK_1_1:
                    # Extrair features manuais dos segmentos filtrados
                    print("\nA extrair features manuais (144 features/janela)...")
                    X_features = []
                    for seg in segmentos_mod_b:
                        fv, _ = extrair_features_janela_4_2(seg, fs=51.2)
                        X_features.append(fv)
                    X_features = np.vstack(X_features)
                    
                    print(f"Features extraídas: {X_features.shape}")
                    analisar_balanceamento_1_1(y_mod_b)
                
                if RUN_META2_TASK_1_3:
                    # Visualização SMOTE para participante 3
                    tarefa_1_3_visualizacao(base_dir=script_dir)
            
            # 5. TAREFA 2: Embeddings Dataset
            if RUN_META2_TASK_2_1:
                print("\n" + "-"*60)
                print("--- TAREFA 2.1: Geração de Embeddings (Modelo Pré-treinado) ---")
                print("-"*60)
                
                # Gerar embeddings dos segmentos filtrados
                X_embeddings = gerar_embeddings_dataset_2_1(segmentos_mod_b, fs_original=51.2)
                
                if X_embeddings.size > 0:
                    print(f"\n✓ Dataset de Embeddings criado com sucesso!")
                    print(f"  Shape: {X_embeddings.shape}")
                    print(f"  Labels: {y_mod_b.shape}")
                    
                    # Estatísticas dos embeddings
                    print(f"\nEstatísticas dos Embeddings:")
                    print(f"  Média: {X_embeddings.mean():.4f}")
                    print(f"  Desvio padrão: {X_embeddings.std():.4f}")
                    print(f"  Min: {X_embeddings.min():.4f}")
                    print(f"  Max: {X_embeddings.max():.4f}")
                    
                    # Opcional: Guardar embeddings para uso posterior
                    try:
                        np.save("meta2_embeddings_X.npy", X_embeddings)
                        np.save("meta2_embeddings_y.npy", y_mod_b)
                        print(f"\n✓ Embeddings guardados em 'meta2_embeddings_X.npy' e 'meta2_embeddings_y.npy'")
                    except Exception as e:
                        print(f"[Aviso] Não foi possível guardar embeddings: {e}")
                else:
                    print("[ERRO] Falha na geração de embeddings.")
            
            # 6. TAREFA 3: Validação e Classificação
            if RUN_META2_TASK_3_1:
                print("\n" + "-"*60)
                print("--- TAREFA 3.1: Split Within-Subject (Train/Val/Test) ---")
                print("-"*60)
                
                # Verificar se temos features e/ou embeddings disponíveis
                if 'X_features' not in locals():
                    print("A extrair features manuais para split...")
                    X_features = []
                    for seg in segmentos_mod_b:
                        fv, _ = extrair_features_janela_4_2(seg, fs=51.2)
                        X_features.append(fv)
                    X_features = np.vstack(X_features)
                
                # Preparar embeddings se existirem
                X_emb_param = X_embeddings if 'X_embeddings' in locals() and X_embeddings.size > 0 else None
                
                # Executar split
                split_data = split_within_subject_3_1(
                    X_features, 
                    y_mod_b, 
                    subjects_mod_b, 
                    X_emb=X_emb_param,
                    train_r=0.6,
                    val_r=0.2,
                    random_state=42
                )
                
                # Acesso aos dados divididos
                print(f"\n✓ Split realizado com sucesso!")
                print(f"  Features Train: {split_data['X_train'].shape}")
                print(f"  Features Val: {split_data['X_val'].shape}")
                print(f"  Features Test: {split_data['X_test'].shape}")
                
                if 'X_emb_train' in split_data:
                    print(f"  Embeddings Train: {split_data['X_emb_train'].shape}")
                    print(f"  Embeddings Val: {split_data['X_emb_val'].shape}")
                    print(f"  Embeddings Test: {split_data['X_emb_test'].shape}")
                
                # Opcional: Guardar splits para uso posterior
                try:
                    np.save("meta2_split_X_train.npy", split_data['X_train'])
                    np.save("meta2_split_y_train.npy", split_data['y_train'])
                    np.save("meta2_split_X_val.npy", split_data['X_val'])
                    np.save("meta2_split_y_val.npy", split_data['y_val'])
                    np.save("meta2_split_X_test.npy", split_data['X_test'])
                    np.save("meta2_split_y_test.npy", split_data['y_test'])
                    
                    if 'X_emb_train' in split_data:
                        np.save("meta2_split_X_emb_train.npy", split_data['X_emb_train'])
                        np.save("meta2_split_X_emb_val.npy", split_data['X_emb_val'])
                        np.save("meta2_split_X_emb_test.npy", split_data['X_emb_test'])
                    
                    print(f"\n✓ Splits guardados em ficheiros .npy")
                except Exception as e:
                    print(f"[Aviso] Não foi possível guardar splits: {e}")
            
            # 7. TAREFA 3.2: Split Between-Subject
            if RUN_META2_TASK_3_2:
                print("\n" + "-"*60)
                print("--- TAREFA 3.2: Split Between-Subject (9/3/3 sujeitos) ---")
                print("-"*60)
                
                # Verificar se temos features e/ou embeddings disponíveis
                if 'X_features' not in locals():
                    print("A extrair features manuais para split...")
                    X_features = []
                    for seg in segmentos_mod_b:
                        fv, _ = extrair_features_janela_4_2(seg, fs=51.2)
                        X_features.append(fv)
                    X_features = np.vstack(X_features)
                
                # Preparar embeddings se existirem
                X_emb_param = X_embeddings if 'X_embeddings' in locals() and X_embeddings.size > 0 else None
                
                # Executar split Between-Subject
                split_data_v2 = split_between_subject_3_2(
                    X_features, 
                    y_mod_b, 
                    subjects_mod_b, 
                    X_emb=X_emb_param,
                    train_subs=9,
                    val_subs=3,
                    test_subs=3,
                    random_state=42
                )
                
                # Acesso aos dados divididos
                print(f"\n✓ Split Between-Subject realizado com sucesso!")
                print(f"  Features Train: {split_data_v2['X_train'].shape} ({len(np.unique(split_data_v2['sub_train']))} sujeitos)")
                print(f"  Features Val: {split_data_v2['X_val'].shape} ({len(np.unique(split_data_v2['sub_val']))} sujeitos)")
                print(f"  Features Test: {split_data_v2['X_test'].shape} ({len(np.unique(split_data_v2['sub_test']))} sujeitos)")
                print(f"  Sujeitos Train: {np.unique(split_data_v2['sub_train'])}")
                print(f"  Sujeitos Val: {np.unique(split_data_v2['sub_val'])}")
                print(f"  Sujeitos Test: {np.unique(split_data_v2['sub_test'])}")
                
                if 'X_emb_train' in split_data_v2:
                    print(f"  Embeddings Train: {split_data_v2['X_emb_train'].shape}")
                    print(f"  Embeddings Val: {split_data_v2['X_emb_val'].shape}")
                    print(f"  Embeddings Test: {split_data_v2['X_emb_test'].shape}")
                
                # Opcional: Guardar splits para uso posterior (com sufixo _v2)
                try:
                    np.save("meta2_split_v2_X_train.npy", split_data_v2['X_train'])
                    np.save("meta2_split_v2_y_train.npy", split_data_v2['y_train'])
                    np.save("meta2_split_v2_X_val.npy", split_data_v2['X_val'])
                    np.save("meta2_split_v2_y_val.npy", split_data_v2['y_val'])
                    np.save("meta2_split_v2_X_test.npy", split_data_v2['X_test'])
                    np.save("meta2_split_v2_y_test.npy", split_data_v2['y_test'])
                    
                    if 'X_emb_train' in split_data_v2:
                        np.save("meta2_split_v2_X_emb_train.npy", split_data_v2['X_emb_train'])
                        np.save("meta2_split_v2_X_emb_val.npy", split_data_v2['X_emb_val'])
                        np.save("meta2_split_v2_X_emb_test.npy", split_data_v2['X_emb_test'])
                    
                    print(f"\n✓ Splits Between-Subject guardados em ficheiros .npy (sufixo _v2)")
                except Exception as e:
                    print(f"[Aviso] Não foi possível guardar splits: {e}")
            
            # 8. TAREFA 3.4: Preparação de Cenários (PCA, ReliefF, etc.)
            if RUN_META2_TASK_3_4:
                print("\n" + "-"*60)
                print("--- TAREFA 3.4: Preparação de Cenários para Classificação ---")
                print("-"*60)
                
                # Determinar qual split usar (prioriza Between-Subject se disponível)
                if 'split_data_v2' in locals():
                    dados_para_processar = split_data_v2
                    print("\nA processar dados da estratégia BETWEEN-SUBJECT (3.2)...")
                elif 'split_data' in locals():
                    dados_para_processar = split_data
                    print("\nA processar dados da estratégia WITHIN-SUBJECT (3.1)...")
                else:
                    print("[Erro] Nenhum split data encontrado. Execute Tarefa 3.1 ou 3.2 primeiro.")
                    dados_para_processar = None
                
                if dados_para_processar:
                    # Executa a pipeline 3.4
                    cenarios_finais = pipeline_preparacao_3_4(dados_para_processar)
                    
                    # Mostrar resumo dos cenários gerados
                    print("\n✓ Cenários gerados e prontos para classificação (Tarefa 4/5).")
                    if 'features' in cenarios_finais:
                        print(f"  Features All: {cenarios_finais['features']['all']['X_train'].shape}")
                        print(f"  Features PCA: {cenarios_finais['features']['pca']['X_train'].shape}")
                        print(f"  Features ReliefF: {cenarios_finais['features']['relief']['X_train'].shape}")
                    
                    if 'embeddings' in cenarios_finais:
                        print(f"  Embeddings All: {cenarios_finais['embeddings']['all']['X_train'].shape}")
                        print(f"  Embeddings PCA: {cenarios_finais['embeddings']['pca']['X_train'].shape}")
                        print(f"  Embeddings ReliefF: {cenarios_finais['embeddings']['relief']['X_train'].shape}")
                    
                    # Exemplo de como aceder aos dados:
                    # X_train_pca = cenarios_finais['features']['pca']['X_train']
                    # y_train = cenarios_finais['features']['pca']['y_train']

    print("\n=== Execução Concluída ===")


if __name__ == "__main__":
    main()