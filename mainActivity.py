# -*- coding: utf-8 -*-
"""
EA/ECAC 2025 - Trabalho Prático 1
mainActivity.py

Implementação completa das Metas 1 e 2.
"""
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessário para plots 3D [cite: 713]
from sklearn.cluster import DBSCAN      # Apenas para o bónus 3.7.1 [cite: 714]

# --- NOVOS IMPORTS PARA A META 2 ---
from scipy import stats as sp_stats      # Para 4.1 (Testes Estatísticos) e 4.2 (Features)
from scipy import fftpack                # Para 4.2 (Features Espectrais)
from scipy.integrate import simps        # Para 4.2 (Features Físicas - Velocidade)
from sklearn.preprocessing import StandardScaler # Para 4.4 (Normalização Z-Score) [cite: 736]
from sklearn.decomposition import PCA    # Para 4.3 (PCA) [cite: 733]
from sklearn.feature_selection import f_classif # Para 4.5 (Fisher Score) [cite: 740]
try:
    # ReliefF não está no sklearn, tente 'skrebate'
    # Instale com: pip install scikit-rebate
    from skrebate import ReliefF         # Para 4.5 (ReliefF) [cite: 740]
except ImportError:
    print("[Aviso Meta 2] Biblioteca 'skrebate' não encontrada. Tarefa 4.5 (ReliefF) falhará.")
    print("Instale com: pip install scikit-rebate")
    ReliefF = None # Define como None para o código não falhar

# --- Constantes Globais (Baseado no PDF) ---
# Mapeamento 1-para-1 com o PDF (1-based-indexing) para 0-based-indexing
COL_DEVICE_ID = 0  # Coluna 1 [cite: 661]
COL_ACC_X = 1      # Coluna 2 [cite: 662]
COL_ACC_Y = 2      # Coluna 3 [cite: 663]
COL_ACC_Z = 3      # Coluna 4 [cite: 664]
COL_GYRO_X = 4     # Coluna 5 [cite: 665]
COL_GYRO_Y = 5     # Coluna 6 [cite: 666]
COL_GYRO_Z = 6     # Coluna 7 [cite: 667]
COL_MAG_X = 7      # Coluna 8 [cite: 668]
COL_MAG_Y = 8      # Coluna 9 [cite: 670]
COL_MAG_Z = 9      # Coluna 10 [cite: 671]
COL_TIMESTAMP = 10 # Coluna 11 [cite: 672]
COL_ACTIVITY = 11  # Coluna 12 [cite: 673]
COL_PARTICIPANT = 12 # Nova coluna (índice 12) adicionada para a Meta 2

# Grupos de colunas para cálculo dos módulos (Meta 1)
MODULO_INDICES = [
    (COL_ACC_X, COL_ACC_Y, COL_ACC_Z),  # Aceleração [cite: 686]
    (COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z), # Giroscópio [cite: 686]
    (COL_MAG_X, COL_MAG_Y, COL_MAG_Z)  # Magnetómetro [cite: 686]
]
# Eixos individuais para Meta 2 (baseado no Artigo 4, Tabela 1 )
META2_EIXOS_INDIVIDUAIS = [
    (COL_ACC_X, "Acel_X"), (COL_ACC_Y, "Acel_Y"), (COL_ACC_Z, "Acel_Z"),
    (COL_GYRO_X, "Gyro_X"), (COL_GYRO_Y, "Gyro_Y"), (COL_GYRO_Z, "Gyro_Z"),
    (COL_MAG_X, "Mag_X"), (COL_MAG_Y, "Mag_Y"), (COL_MAG_Z, "Mag_Z")
]


# Labels para gráficos
VAR_LABELS = [
    "Módulo Acelerômetro",
    "Módulo Giroscópio",
    "Módulo Magnetômetro"
]
SENSOR_LABELS = {
    1: "Pulso Esquerdo",  # ID 1 [cite: 677]
    2: "Pulso Direito",   # ID 2 [cite: 677]
    3: "Peito",           # ID 3 [cite: 677]
    4: "Perna Sup. Direita", # ID 4 [cite: 677]
    5: "Perna Inf. Esquerda" # ID 5 [cite: 677]
}

# =============================================================================
# --- META 1: TAREFA 2 e 3 (O SEU CÓDIGO ORIGINAL)
# =============================================================================

# --- Tarefa 2: Carregamento de Dados ---

def carregar_dados_participante(num_participante, base_dir="."):
    """
    (Tarefa 2) Lê todos os 5 ficheiros CSV do participante e devolve um array NumPy. [cite: 683]
    """
    part_dir = os.path.join(base_dir, f"part{num_participante}")
    dados = []

    for i in range(1, 6): # Dispositivos 1 a 5 [cite: 656]
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
    (Helper) Carrega dados de TODOS os 15 participantes. [cite: 649]
    [MODIFICADO PARA META 2]: Adiciona uma coluna com o ID do participante.
    """
    dados_todos = []
    # O dataset tem 15 participantes (part0 a part14) [cite: 649]
    for part_num in range(15): 
        print(f"A carregar participante {part_num}...")
        dados_participante = carregar_dados_participante(part_num, base_dir)
        
        if dados_participante.size > 0:
            # --- MODIFICAÇÃO CHAVE (META 2) ---
            # Cria uma coluna com o ID do participante (part_num)
            # com o mesmo número de linhas que dados_participante
            id_participante_col = np.full((dados_participante.shape[0], 1), part_num)
            
            # Junta a nova coluna aos dados
            # Shape final: (n_amostras, 13)
            dados_com_id = np.hstack((dados_participante, id_participante_col))
            dados_todos.append(dados_com_id)
            # --- FIM DA MODIFICAÇÃO ---
    
    if not dados_todos:
        print("[Erro Fatal] Nenhum dado carregado. Verifique o caminho `base_dir`.")
        return np.array([])
        
    return np.concatenate(dados_todos, axis=0)

# --- Tarefa 3 (Intro): Preparação de Dados ---

def calcular_modulo_vetor(dados, col_x, col_y, col_z):
    """
    Calcula o módulo (norma Euclidiana) de um vetor 3D.
    Fórmula: ||t|| = sqrt(tx^2 + ty^2 + tz^2) [cite: 689]
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

# --- Tarefa 3.1: Boxplot de Atividades ---

def plotar_boxplots_3_1(dados_transformados):
    """
    (Tarefa 3.1) Gera 15 boxplots (3 vars x 5 sensores). [cite: 690]
    Eixo X: Atividade (1-16) [cite: 690]
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
            dados_box = []
            for a in range(1, 17): # Atividades 1 a 16 [cite: 674]
                activity_mask = dados_sensor[:, 1] == a
                dados_ativ = dados_sensor[activity_mask, var_idx + 2] # +2 para aceder às cols de módulo
                dados_box.append(dados_ativ)
                
            # 3. Plotar
            ax.boxplot(dados_box, labels=[str(a) for a in range(1, 17)])

            # 4. Legendas
            if var_idx == 0:
                ax.set_title(SENSOR_LABELS[sensor_id], fontsize=14, fontweight='bold')
            if sensor_id == 1:
                ax.set_ylabel(VAR_LABELS[var_idx], fontsize=14, fontweight='bold')
            
            ax.set_xlabel("Atividade")
            ax.tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("meta1_tarefa_3_1_boxplots.png")
    print("Gráfico 'meta1_tarefa_3_1_boxplots.png' guardado.")

# --- Tarefa 3.2: Análise Densidade de Outliers (IQR) ---

def analisar_densidade_iqr_3_2(dados_transformados):
    """
    (Tarefa 3.2) Analisa e comenta a densidade de outliers. [cite: 693]
    Usa método IQR (Tukey). [cite: 694]
    Filtra apenas para o Pulso Direito (ID 2). [cite: 694]
    """
    print("\n--- Início Tarefa 3.2: Densidade de Outliers (IQR - Pulso Direito) ---")
    
    sensor_id_foco = 2 # Pulso Direito [cite: 694]
    
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
            
            n_r = len(amostras) # nr = número total de pontos [cite: 700]
            if n_r == 0:
                continue # Pula atividades sem dados

            # 3. Aplicar método IQR (Tukey) [cite: 694]
            q1 = np.percentile(amostras, 25)
            q3 = np.percentile(amostras, 75)
            iqr = q3 - q1
            
            limite_inf = q1 - (1.5 * iqr)
            limite_sup = q3 + (1.5 * iqr)
            
            # 4. Contar outliers (no)
            outliers_mask = (amostras < limite_inf) | (amostras > limite_sup)
            n_o = np.sum(outliers_mask) # no = número de outliers [cite: 700]
            
            # 5. Calcular densidade (d) [cite: 696]
            densidade = (n_o / n_r) * 100
            
            print(f"| {var_label:21} | {a:9} | {n_r:12} | {n_o:15} | {densidade:13.2f}% |")

    print("\n--- Fim Tarefa 3.2 ---")

# --- Tarefa 3.3: Rotina Z-Score ---

def identificar_outliers_zscore_3_3(amostras, k):
    """
    (Tarefa 3.3) Identifica outliers usando Z-Score para um k variável. [cite: 701]
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

# --- Tarefa 3.4: Plots Z-Score ---

def plotar_outliers_zscore_3_4(dados_transformados):
    """
    (Tarefa 3.4) Gera plots análogos a 3.1, mas usando Z-Score. [cite: 703]
    Outliers a vermelho, inliers a azul. [cite: 704]
    Testa k = 3, 3.5 e 4. [cite: 705]
    """
    print("A gerar gráficos para Tarefa 3.4...")
    
    k_valores = [3, 3.5, 4]  # [cite: 705]
    
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
        plt.savefig(f"meta1_tarefa_3_4_zscore_k{k}.png")
        print(f"Gráfico 'meta1_tarefa_3_4_zscore_k{k}.png' guardado.")

# --- Tarefa 3.5: Comparação IQR vs Z-Score ---

def comparar_densidades_3_5(dados_transformados):
    """
    (Tarefa 3.5) Compara e discute resultados de 3.1 (IQR) e 3.4 (Z-Score). [cite: 708]
    Foco: Apenas sensores do pulso direito (ID 2). [cite: 708]
    """
    print("\n--- Início Tarefa 3.5: Comparação Densidades (IQR vs Z-Score) - Pulso Direito ---")
    
    sensor_id_foco = 2 # Pulso Direito [cite: 708]
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

    print("\n--- Discussão (Tarefa 3.5) ---")
    print("1. O método IQR (usado em 3.1/3.2) é robusto a distribuições não-normais.")
    print("2. O método Z-Score (3.3/3.4) assume uma distribuição 'normal' (gaussiana) e é sensível a outliers (que afetam a média e o desvio padrão).")
    print("3. Comparação: A tabela mostra que as densidades diferem. Para atividades dinâmicas (ex: 6, 7), o IQR tende a ser mais 'agressivo' (encontra mais outliers) do que o Z-Score (k=3).")
    print("4. Efeito do 'k': Como esperado, aumentar o 'k' (de 3 para 4) [cite: 705] torna o Z-Score mais permissivo, diminuindo a densidade de outliers detetada.")
    print("--- Fim Tarefa 3.5 ---")

# --- Tarefa 3.6: Rotina K-Means ---

def kmeans_3_6(X, n_clusters, max_iter=100, tol=1e-4):
    """
    (Tarefa 3.6) Implementa o algoritmo k-means. [cite: 709]
    """
    # 1. Inicialização: Escolhe n_clusters pontos aleatórios dos dados como centroides
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroides = X[indices]
    
    for _ in range(max_iter):
        # 2. Atribuição: Calcula distâncias de *todos* os pontos a *todos* os centroides
        distancias = np.sqrt(((X[:, np.newaxis] - centroides) ** 2).sum(axis=2))
        
        #    Encontra o índice do centroide mais próximo (axis=1)
        labels = np.argmin(distancias, axis=1)
        
        # 3. Atualização: Calcula o novo centroide (média) para cada cluster
        novos_centroides = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # 4. Verificação de Convergência
        if np.all(np.linalg.norm(novos_centroides - centroides, axis=1) < tol):
            break
            
        centroides = novos_centroides
        
    return centroides, labels

# --- Tarefa 3.7: Outliers com K-Means e Plot 3D ---

def analisar_outliers_kmeans_3_7(dados, n_clusters_lista):
    """
    (Tarefa 3.7) Determina outliers usando k-means no espaço original (x,y,z). [cite: 711]
    Gera plots 3D. [cite: 713]
    """
    print(f"\n--- Início Tarefa 3.7: Outliers K-Means (n_clusters={n_clusters_lista}) ---")
    
    # Foco: Aceleração (var 0), Pulso Direito (sensor 2), Atividade 4 (Walk)
    var_idx_foco = 0 # Aceleração
    sensor_id_foco = 2 # Pulso Direito
    atividade_foco = 4 # Walk [cite: 674]
    
    cols_xyz = MODULO_INDICES[var_idx_foco]
    
    # 1. Filtrar dados (usando 'dados' raw)
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
            centroides, labels = kmeans_3_6(X, n_clusters)
        except ValueError as e:
            print(f"    Erro no K-Means (provavelmente cluster vazio): {e}")
            continue

        # 3. Estratégia de Outlier:
        distancias_ao_centroide = np.linalg.norm(X - centroides[labels], axis=1)
        # Define outliers como pontos no 99º percentil de distância
        limite_dist = np.percentile(distancias_ao_centroide, 99)
        outlier_mask = distancias_ao_centroide > limite_dist
        inlier_mask = ~outlier_mask
        
        n_o = np.sum(outlier_mask)
        print(f"    Encontrados {n_o} outliers ({n_o/X.shape[0]*100:.2f}%)")

        # 4. Plot 3D [cite: 713]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2],
            c=labels[inlier_mask], cmap='tab10', alpha=0.6, s=10, label="Inliers (Clusters)"
        )
        ax.scatter(
            X[outlier_mask, 0], X[outlier_mask, 1], X[outlier_mask, 2],
            c='red', s=50, label="Outliers"
        )
        ax.scatter(
            centroides[:, 0], centroides[:, 1], centroides[:, 2],
            c='black', s=200, marker='X', label="Centroides"
        )

        ax.set_xlabel("Acelerômetro X")
        ax.set_ylabel("Acelerômetro Y")
        ax.set_zlabel("Acelerômetro Z")
        ax.set_title(
            f"Tarefa 3.7: K-Means (k={n_clusters}) - Sensor '{SENSOR_LABELS[sensor_id_foco]}' - Atividade {atividade_foco}"
        )
        ax.legend()
        plt.savefig(f"meta1_tarefa_3_7_kmeans_k{n_clusters}.png")
        print(f"Gráfico 'meta1_tarefa_3_7_kmeans_k{n_clusters}.png' guardado.")

    print("\n--- Discussão (Comparação 3.7 vs 3.4) ---")
    print("1. O K-Means (3.7) é 'multivariado'. Encontra outliers no espaço 3D (x,y,z). [cite: 711]")
    print("2. O Z-Score (3.4) foi 'univariado'. Analisou o *módulo* (um único número) de cada vez. [cite: 703]")
    print("3. Diferença Chave: Um ponto pode ser um outlier 3D sem ser um outlier 1D (Z-Score).")
    print("   Ex: (x=10, y=10, z=0) pode ter um módulo normal, mas ser uma combinação (x,y,z) muito rara.")
    print("--- Fim Tarefa 3.7 ---")

# --- Tarefa 3.7.1 (Bónus): Outliers com DBSCAN ---

def analisar_outliers_dbscan_3_7_1(dados):
    """
    (Tarefa 3.7.1) Determina outliers usando DBSCAN. [cite: 714]
    """
    print(f"\n--- Início Tarefa 3.7.1 (Bónus): Outliers DBSCAN ---")
    
    # Usar a mesma fatia de dados da Tarefa 3.7
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

    # 1. Correr DBSCAN [cite: 714]
    db = DBSCAN(eps=0.5, min_samples=10).fit(X)
    
    # 2. Outliers são pontos com label = -1
    outlier_mask = db.labels_ == -1
    inlier_mask = ~outlier_mask
    
    n_o = np.sum(outlier_mask)
    print(f"  DBSCAN (eps=0.5, min=10) encontrou {n_o} outliers ({n_o/X.shape[0]*100:.2f}%)")

    # 3. Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X[inlier_mask, 0], X[inlier_mask, 1], X[inlier_mask, 2],
        c=db.labels_[inlier_mask], cmap='tab10', alpha=0.6, s=10, label="Inliers (Clusters)"
    )
    ax.scatter(
        X[outlier_mask, 0], X[outlier_mask, 1], X[outlier_mask, 2],
        c='red', s=50, label="Outliers (Ruído)"
    )

    ax.set_xlabel("Acelerômetro X")
    ax.set_ylabel("Acelerômetro Y")
    ax.set_zlabel("Acelerômetro Z")
    ax.set_title(f"Tarefa 3.7.1: DBSCAN - Sensor '{SENSOR_LABELS[sensor_id_foco]}' - Atividade {atividade_foco}")
    ax.legend()
    plt.savefig("meta1_tarefa_3_7_1_dbscan.png")
    print("Gráfico 'meta1_tarefa_3_7_1_dbscan.png' guardado.")
    print("--- Fim Tarefa 3.7.1 ---")


# =============================================================================
# --- META 2: TAREFA 4 - EXTRAÇÃO DE FEATURES E SELEÇÃO
# =============================================================================

# --- Tarefa 4.1: Significância Estatística ---

def analisar_significancia_estatistica_4_1(dados_transformados):
    """
    (Tarefa 4.1) Analisa a significância estatística dos valores médios
    nas diferentes atividades. [cite: 717]
    """
    print(f"\n--- Início Tarefa 4.1: Significância Estatística ---")
    
    # Foco: Pulso Direito (ID 2), como em 3.2 [cite: 694]
    sensor_id_foco = 2
    sensor_mask = dados_transformados[:, 0] == sensor_id_foco
    dados_sensor = dados_transformados[sensor_mask]

    for var_idx in range(3): # Acel, Gyro, Mag
        var_label = VAR_LABELS[var_idx]
        print(f"\nAnalisando: {var_label} - Sensor {SENSOR_LABELS[sensor_id_foco]}")
        
        # 1. Obter os dados (módulos) para cada atividade
        dados_por_atividade = []
        for a in range(1, 17):
            activity_mask = dados_sensor[:, 1] == a
            amostras = dados_sensor[activity_mask, var_idx + 2]
            if amostras.size > 20: # Só considera se tiver dados suficientes
                dados_por_atividade.append(amostras)
            else:
                dados_por_atividade.append(np.array([])) # Placeholder

        # 2. Testar Normalidade (sugerido KS[cite: 723], usamos 'normaltest' que é mais direto)
        # Vamos testar a Atividade 1 (Stand)
        ativ_1_data = dados_por_atividade[0]
        if ativ_1_data.size > 20:
            stat, p_norm = sp_stats.normaltest(ativ_1_data)
            is_normal = p_norm > 0.05
            print(f"  Teste de Normalidade (p-value): {p_norm:.2e} (Normal: {is_normal})")
        else:
            is_normal = False # Assume não-normal se não houver dados
            print("  Teste de Normalidade: Dados insuficientes, assumindo não-normal.")

        # 3. Teste de Significância (ANOVA ou Kruskal-Wallis)
        # O objetivo é ver se a média é ESTATISTICAMENTE diferente entre as 16 atividades [cite: 717]
        dados_validos = [d for d in dados_por_atividade if d.size > 20]
        
        if len(dados_validos) < 2:
            print("  Teste de Significância: Menos de 2 grupos de atividades com dados. A saltar.")
            continue
            
        if is_normal:
            # Se fossem normais, usaríamos ANOVA (teste paramétrico)
            f_stat, p_value = sp_stats.f_oneway(*dados_validos)
            print(f"  Teste Significância (ANOVA - Paramétrico):")
        else:
            # Como (provavelmente) não são normais, usamos Kruskal-Wallis (não-paramétrico)
            h_stat, p_value = sp_stats.kruskal(*dados_validos)
            print(f"  Teste Significância (Kruskal-Wallis - Não-Paramétrico):")

        print(f"    p-value: {p_value:.2e}")
        if p_value < 0.05:
            print(f"    Conclusão: (p < 0.05) Os valores médios de '{var_label}' SÃO estatisticamente diferentes entre as atividades.")
        else:
            print(f"    Conclusão: (p > 0.05) NÃO HÁ diferença estatística significativa.")
            
    print("--- Fim Tarefa 4.1 ---")


# --- Tarefa 4.2: Extração de Features (Temporal e Espectral) ---

# --- Funções Helper para a Tarefa 4.2 (Baseadas no Artigo 4) ---

def _calc_zcr(data):
    """ Calcula Zero Crossing Rate [cite: 329] """
    return np.sum(np.diff(np.sign(data)) != 0) / (len(data) - 1)

def _calc_mcr(data):
    """ Calcula Mean Crossing Rate [cite: 329] """
    mean = np.mean(data)
    return np.sum(np.diff(np.sign(data - mean)) != 0) / (len(data) - 1)

def _calc_spectral_features(data, fs):
    """ 
    Calcula features espectrais: Entropia, Freq. Dominante, Energia 
    Baseado no Artigo 4 (Tabela 1 [cite: 329] e Sec 3.3 [cite: 361, 362])
    """
    n_samples = len(data)
    if n_samples < 2:
        return 0, 0, 0
        
    # Calcula FFT
    yf = fftpack.fft(data)
    # Calcula Frequências (positivas)
    xf = fftpack.fftfreq(n_samples, 1/fs)[:n_samples//2]
    # Calcula Power Spectrum Density (PSD)
    psd = (2.0/n_samples) * np.abs(yf[0:n_samples//2])**2
    
    # 1. Energia (soma dos quadrados das magnitudes FFT, exceto DC) [cite: 362, 370]
    #    (Usamos o PSD que já está ao quadrado e normalizado)
    energy = np.sum(psd)
    
    # 2. Frequência Dominante [cite: 361]
    dominant_freq = xf[np.argmax(psd)] if psd.size > 0 else 0
    
    # 3. Entropia Espectral [cite: 329]
    psd_norm = psd / np.sum(psd) # Normaliza para ser uma distribuição de prob.
    # Evita log(0)
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    
    return spectral_entropy, dominant_freq, energy

def _calc_velocidade_features(window_data, fs):
    """
    Calcula features de velocidade (AVH, AVG) [cite: 355, 356]
    window_data: Apenas dados do Acelerómetro (N, 3)
    """
    n_samples = window_data.shape[0]
    time_axis = np.linspace(0, (n_samples-1)/fs, n_samples)
    
    # Integra Aceleração para obter Velocidade
    vel_x = simps(window_data[:, 0], time_axis)
    vel_y = simps(window_data[:, 1], time_axis)
    vel_z = simps(window_data[:, 2], time_axis)
    
    # Assume z = gravidade, (x,y) = plano horizontal (interpretação)
    # 6. Averaged Velocity along Heading Direction (AVH) [cite: 355]
    avh = np.sqrt(vel_x**2 + vel_y**2) / (n_samples / fs)
    
    # 7. Averaged Velocity along Gravity Direction (AVG) [cite: 356]
    avg = vel_z / (n_samples / fs)
    
    return avh, avg

def extrair_features_4_2(dados_raw, fs=50.0):
    """
    (Tarefa 4.2) Extrai o feature set temporal e espectral. [cite: 724]
    Implementa 'sliding window' de 5s com 50% overlap. 
    
    `dados_raw` deve ser o array com a coluna de PARTICIPANTE (12).
    """
    print(f"\n--- Início Tarefa 4.2: Extração de Features ---")
    
    # Parâmetros da Janela (Conforme TP1.pdf)
    window_size_sec = 5.0 # 
    overlap_ratio = 0.5   # 
    
    window_samples = int(fs * window_size_sec)
    step_samples = int(window_samples * (1 - overlap_ratio))
    
    X_features_list = [] # O nosso 'X' (a matriz de features)
    y_labels_list = []   # O nosso 'y' (as etiquetas/labels)
    feature_names = []   # Nomes das features (para 4.6)
    
    total_janelas = 0

    # É CRUCIAL segmentar os dados antes de aplicar a janela,
    # para não misturar participantes ou atividades numa só janela. [cite: 727]
    
    participantes = np.unique(dados_raw[:, COL_PARTICIPANT])
    atividades = np.unique(dados_raw[:, COL_ACTIVITY])
    sensores = np.unique(dados_raw[:, COL_DEVICE_ID])
    
    print(f"A processar {len(participantes)} participantes, {len(sensores)} sensores, {len(atividades)} atividades...")
    print(f"Janela: {window_size_sec}s ({window_samples} amostras), Overlap: {overlap_ratio*100}% ({step_samples} amostras)")

    for part_id in participantes:
        print(f"  Processando Part. {int(part_id)}...")
        for sensor_id in sensores:
            for act_id in atividades:
                
                # 1. Isolar o segmento de dados contínuo
                mask = (dados_raw[:, COL_PARTICIPANT] == part_id) & \
                       (dados_raw[:, COL_DEVICE_ID] == sensor_id) & \
                       (dados_raw[:, COL_ACTIVITY] == act_id)
                
                segmento = dados_raw[mask]
                
                # Ordenar por Timestamp (vital)
                segmento = segmento[segmento[:, COL_TIMESTAMP].argsort()]
                
                # 2. Aplicar a janela deslizante (sliding window)
                idx = 0
                while (idx + window_samples) <= segmento.shape[0]:
                    
                    # Extrai a janela de dados raw (13 colunas)
                    window_full = segmento[idx : idx + window_samples]
                    
                    # --- VALIDAÇÃO (Nova) ---
                    # O enunciado [cite: 727] diz: "Se um segmento cobrir mais do que uma atividade, descarte".
                    # A nossa lógica de loop já faz isto, mas vamos re-confirmar.
                    if np.unique(window_full[:, COL_ACTIVITY]).size > 1:
                        # Este caso não deve acontecer devido ao loop, mas é uma boa salvaguarda
                        idx += step_samples
                        continue
                    
                    features_janela = []
                    
                    # --- A. FEATURES ESTATÍSTICAS (Artigo 4, Tabela 1 ) ---
                    # Aplicadas a todos os 9 eixos
                    for col_idx, name in META2_EIXOS_INDIVIDUAIS:
                        data = window_full[:, col_idx]
                        
                        features_janela.extend([
                            np.mean(data),             # 1. Mean [cite: 329]
                            np.median(data),           # 2. Median [cite: 329]
                            np.std(data),              # 3. Standard Deviation [cite: 329]
                            np.var(data),              # 4. Variance [cite: 329]
                            np.sqrt(np.mean(data**2)), # 5. Root Mean Square [cite: 329]
                            np.mean(np.diff(data)),    # 6. Averaged derivatives [cite: 329]
                            sp_stats.skew(data),       # 7. Skewness [cite: 329]
                            sp_stats.kurtosis(data),   # 8. Kurtosis [cite: 329]
                            sp_stats.iqr(data),        # 9. Interquartile Range [cite: 329]
                            _calc_zcr(data),           # 10. Zero Crossing Rate [cite: 329]
                            _calc_mcr(data)            # 11. Mean Crossing Rate [cite: 329]
                        ])
                        
                        # Features Espectrais (por eixo)
                        s_ent, d_freq, s_en = _calc_spectral_features(data, fs)
                        features_janela.extend([
                            s_ent,  # 12. Spectral Entropy [cite: 329]
                            d_freq, # (Física 8. Dominant Frequency) [cite: 361]
                            s_en    # (Física 9. Energy) [cite: 362]
                        ])
                        
                        if total_janelas == 0: # Só guarda nomes na 1ª janela
                            feature_names.extend([
                                f"{name}_mean", f"{name}_median", f"{name}_std", f"{name}_var",
                                f"{name}_rms", f"{name}_avg_deriv", f"{name}_skew", f"{name}_kurt",
                                f"{name}_iqr", f"{name}_zcr", f"{name}_mcr", f"{name}_spec_ent",
                                f"{name}_dom_freq", f"{name}_spec_energy"
                            ])
                    
                    # --- B. FEATURES ESTATÍSTICAS (Correlação) [cite: 329] ---
                    # (Dentro de cada sensor)
                    for (c_x, c_y, c_z, name) in [(COL_ACC_X, COL_ACC_Y, COL_ACC_Z, "Acel"),
                                                  (COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z, "Gyro"),
                                                  (COL_MAG_X, COL_MAG_Y, COL_MAG_Z, "Mag")]:
                        cor_xy = np.corrcoef(window_full[:, c_x], window_full[:, c_y])[0, 1]
                        cor_xz = np.corrcoef(window_full[:, c_x], window_full[:, c_z])[0, 1]
                        cor_yz = np.corrcoef(window_full[:, c_y], window_full[:, c_z])[0, 1]
                        features_janela.extend([cor_xy, cor_xz, cor_yz])
                        
                        if total_janelas == 0:
                            feature_names.extend([f"Corr_{name}_XY", f"Corr_{name}_XZ", f"Corr_{name}_YZ"])
                    
                    # --- C. FEATURES FÍSICAS (Artigo 4, Sec 3.3) ---
                    # (Apenas Acel/Gyro, conforme definido no artigo)
                    win_acel = window_full[:, [COL_ACC_X, COL_ACC_Y, COL_ACC_Z]]
                    
                    # 1. Movement Intensity (AI, VI)
                    mi_timeseries = np.linalg.norm(win_acel, axis=1)
                    feat_ai = np.mean(mi_timeseries) # [cite: 335]
                    feat_vi = np.var(mi_timeseries)  # [cite: 339]
                    
                    # 2. Signal Magnitude Area (SMA) [cite: 343]
                    feat_sma = (np.sum(np.abs(win_acel[:, 0])) + 
                                np.sum(np.abs(win_acel[:, 1])) + 
                                np.sum(np.abs(win_acel[:, 2]))) / window_samples
                    
                    # 3. Eigenvalues (EVA) [cite: 351]
                    cov_matrix = np.cov(win_acel.T)
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    eigenvalues.sort() # Ordena do menor para o maior
                    feat_eva1, feat_eva2 = eigenvalues[-1], eigenvalues[-2] # Top 2
                    
                    # 4. Correlation Acel Grav/Heading (CAGH) [cite: 354]
                    # (Assumindo Z=Grav, (X,Y)=Heading)
                    norm_heading = np.linalg.norm(win_acel[:, [0, 1]], axis=1)
                    feat_cagh = np.corrcoef(win_acel[:, 2], norm_heading)[0, 1]
                    
                    # 5, 6. Velocidade (AVH, AVG) [cite: 355, 356]
                    feat_avh, feat_avg = _calc_velocidade_features(win_acel, fs)

                    # 7. Averaged Rotation (ARATG) [cite: 358]
                    # (Média da rotação no eixo Z do Giroscópio)
                    feat_aratg = np.mean(window_full[:, COL_GYRO_Z])
                    
                    # 10, 11. Averaged Energy (AAE, ARE)
                    # (As 'Energies' individuais já foram calculadas)
                    # Índices: Acel (13, 27, 41), Gyro (55, 69, 83)
                    # (Isto é frágil, vamos recalcular)
                    s_en_ax = features_janela[feature_names.index("Acel_X_spec_energy")]
                    s_en_ay = features_janela[feature_names.index("Acel_Y_spec_energy")]
                    s_en_az = features_janela[feature_names.index("Acel_Z_spec_energy")]
                    s_en_gx = features_janela[feature_names.index("Gyro_X_spec_energy")]
                    s_en_gy = features_janela[feature_names.index("Gyro_Y_spec_energy")]
                    s_en_gz = features_janela[feature_names.index("Gyro_Z_spec_energy")]
                    
                    feat_aae = (s_en_ax + s_en_ay + s_en_az) / 3.0 # [cite: 371]
                    feat_are = (s_en_gx + s_en_gy + s_en_gz) / 3.0 # [cite: 372]
                    
                    features_janela.extend([
                        feat_ai, feat_vi, feat_sma, feat_eva1, feat_eva2,
                        feat_cagh, feat_avh, feat_avg, feat_aratg,
                        feat_aae, feat_are
                    ])

                    if total_janelas == 0:
                        feature_names.extend([
                            "Phys_AI", "Phys_VI", "Phys_SMA", "Phys_EVA1", "Phys_EVA2",
                            "Phys_CAGH", "Phys_AVH", "Phys_AVG", "Phys_ARATG",
                            "Phys_AAE", "Phys_ARE"
                        ])

                    # --- Fim da Extração ---
                    
                    # Guardar features e label
                    X_features_list.append(features_janela)
                    y_labels_list.append(act_id) # A label desta janela
                    
                    # Avançar a janela
                    idx += step_samples
                    total_janelas += 1

    print(f"Extração concluída. Total de {total_janelas} janelas (vetores de features) criadas.")
    print(f"Número total de features extraídas: {len(feature_names)}")
    
    if not X_features_list:
        print("[ERRO] Nenhuma feature extraída. Verifique os parâmetros da janela.")
        return np.array([]), np.array([]), []
        
    return np.array(X_features_list), np.array(y_labels_list), feature_names


# --- Tarefa 4.3 (PCA) e 4.4 (Análise PCA) ---

def analisar_pca_4_3_4_4(X_features):
    """
    (Tarefa 4.3/4.4) Implementa e analisa o PCA. [cite: 733, 735]
    X_features: A matriz (X) de features extraída da Tarefa 4.2.
    """
    print(f"\n--- Início Tarefa 4.3/4.4: Análise PCA ---")
    if X_features.size == 0:
        print("Dataset de features vazio. A saltar PCA.")
        return

    # 1. Normalizar as features usando Z-Score [cite: 736]
    print("  1. A normalizar features (Z-Score)...")
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_features)
    
    # 2. Implementar PCA [cite: 733]
    print("  2. A calcular PCA...")
    pca = PCA() # Sem n_components, calcula todos
    pca.fit(X_norm)
    
    # 3. Determinar importância (Variância Explicada) [cite: 735]
    print("  3. A analisar variância explicada...")
    # Variância *acumulada*
    var_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # 4. Quantas dimensões para 75%? [cite: 736]
    n_componentes_75 = np.argmax(var_acumulada >= 0.75) + 1
    print(f"    -> São necessárias {n_componentes_75} dimensões para explicar 75% da variância.")

    # 5. Plot (Scree Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(var_acumulada) + 1), var_acumulada, marker='o', linestyle='--')
    plt.axhline(y=0.75, color='r', linestyle=':', label='75% Limite [cite: 736]')
    plt.axvline(x=n_componentes_75, color='g', linestyle=':', label=f'{n_componentes_75} Componentes')
    plt.title('Tarefa 4.4: Variância Acumulada Explicada pelo PCA')
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Variância Acumulada Explicada')
    plt.legend()
    plt.grid(True)
    plt.savefig("meta2_tarefa_4_4_pca.png")
    print("Gráfico 'meta2_tarefa_4_4_pca.png' guardado.")
    
    # 6. Tarefa 4.4.1: Obter features comprimidas (exemplo) [cite: 737]
    print("\n  (4.4.1) Exemplo de compressão:")
    pca_75 = PCA(n_components=n_componentes_75)
    X_comprimido = pca_75.fit_transform(X_norm)
    print(f"    Dimensão original: {X_norm.shape}")
    print(f"    Dimensão comprimida (75%): {X_comprimido.shape}")
    print(f"    Exemplo da 1ª amostra (comprimida): {X_comprimido[0]}")
    
    # 7. Tarefa 4.4.2: Vantagens e Limitações [cite: 738]
    print("\n  (4.4.2) Vantagens e Limitações do PCA:")
    print("    Vantagens:")
    print("      - Reduz a 'maldição da dimensionalidade' (melhora performance de ML).")
    print("      - Remove redundância (features correlacionadas) criando componentes ortogonais.")
    print("      - Não supervisionado (não precisa de labels).")
    print("    Limitações:")
    print("      - Perda de interpretabilidade (componentes são 'misturas' de features originais).")
    print("      - Assume que a variância = importância, o que nem sempre é verdade para classificação.")
    print("      - Sensível à escala dos dados (requer normalização Z-Score, como fizemos [cite: 736]).")
    
    print("--- Fim Tarefa 4.4 ---")


# --- Tarefa 4.5 (Seleção) e 4.6 (Análise) ---

def analisar_selecao_features_4_5_4_6(X_features, y_labels, feature_names):
    """
    (Tarefa 4.5/4.6) Implementa Fisher Score e ReliefF. [cite: 740]
    X_features: A matriz (X) de features (NÃO normalizada)
    y_labels: O vetor (y) de etiquetas (atividades)
    """
    print(f"\n--- Início Tarefa 4.5/4.6: Seleção (Fisher e ReliefF) ---")
    if X_features.size == 0 or y_labels.size == 0:
        print("Dataset de features vazio. A saltar seleção.")
        return
        
    # Garantir que não há NaNs/Infs (ReliefF falha)
    if not np.all(np.isfinite(X_features)):
        print("  Aviso: Encontrados NaNs/Infs nas features. Substituindo por 0.")
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Fisher Feature Score [cite: 740]
    # (Vamos usar ANOVA F-value, que é a base do Fisher Score para multiclasse)
    print("  1. A calcular Fisher Score (ANOVA F-value)...")
    f_scores, _ = f_classif(X_features, y_labels)
    # Obter os índices das 10 melhores [cite: 742]
    top_10_fisher_indices = np.argsort(f_scores)[::-1][:10]
    top_10_fisher_features = [feature_names[i] for i in top_10_fisher_indices]
    
    print("\n    --- Top 10 Features (Fisher Score) ---")
    for i, (idx) in enumerate(top_10_fisher_indices):
        print(f"    {i+1}. {feature_names[idx]} (Score: {f_scores[idx]:.2f})")
    
    # 2. ReliefF [cite: 740]
    print("\n  2. A calcular ReliefF...")
    top_10_relief_features = []
    if ReliefF is not None:
        # ReliefF é computacionalmente caro. Pode demorar.
        # Normalização Z-Score é recomendada para ReliefF
        scaler = StandardScaler()
        X_norm_relief = scaler.fit_transform(X_features)
        
        fs_relief = ReliefF(n_neighbors=10)
        fs_relief.fit(X_norm_relief, y_labels)
        
        # Obter os scores e os top 10 [cite: 742]
        relief_scores = fs_relief.feature_importances_
        top_10_relief_indices = np.argsort(relief_scores)[::-1][:10]
        top_10_relief_features = [feature_names[i] for i in top_10_relief_indices]

        print("\n    --- Top 10 Features (ReliefF) ---")
        for i, (idx) in enumerate(top_10_relief_indices):
            print(f"    {i+1}. {feature_names[idx]} (Score: {relief_scores[idx]:.4f})")
        
        # 3. Comparação [cite: 742]
        print("\n  3. Comparação:")
        comuns = set(top_10_fisher_features) & set(top_10_relief_features)
        print(f"    Features comuns no Top 10: {list(comuns)}")

    else:
        print("    ReliefF não pôde ser executado (skrebate não instalado).")
        top_10_relief_indices = []

    # 4. Tarefa 4.6.1: Obter features comprimidas (exemplo) [cite: 748]
    print("\n  (4.6.1) Exemplo de seleção (usando Fisher):")
    X_selecionado_fisher = X_features[:, top_10_fisher_indices]
    print(f"    Dimensão original: {X_features.shape}")
    print(f"    Dimensão comprimida (Fisher Top 10): {X_selecionado_fisher.shape}")
    print(f"    Exemplo da 1ª amostra (selecionada): {X_selecionado_fisher[0]}")

    # 5. Tarefa 4.6.2: Vantagens e Limitações [cite: 749]
    print("\n  (4.6.2) Vantagens e Limitações (Seleção vs PCA):")
    print("    Vantagens (Seleção de Features):")
    print("      - MANTÉM a interpretabilidade (sabemos que 'Acel_X_mean' é a média do Acel-X).")
    print("      - Geralmente mais rápido de computar que PCA (depende do método).")
    print("      - O modelo final é mais simples (usa menos inputs).")
    print("    Limitações (Seleção de Features):")
    print("      - IGNORA a redundância (pode selecionar 2 features que são 99% correlacionadas).")
    print("      - Fisher Score: Só vê cada feature isoladamente (univariado).")
    print("      - ReliefF: É melhor (multivariado, capta interações), mas muito caro de computar.")

    print("--- Fim Tarefa 4.6 ---")


# =============================================================================
# --- FUNÇÃO PRINCIPAL (main)
# =============================================================================

def main():
    """
    Função principal que orquestra a execução das tarefas da Meta 1 e 2.
    """
    
    # --- PAINEL DE CONTROLO ---
    # Defina como True/False para executar/saltar cada tarefa
    
    # === META 1 ===
    RUN_TASK_2_TEST = False 
    RUN_TASK_3_1 = False
    RUN_TASK_3_2 = False
    RUN_TASK_3_4 = False
    RUN_TASK_3_5 = False
    RUN_TASK_3_7 = False
    RUN_TASK_3_7_1 = False
    
    # === META 2 ===
    # (Executa 4.1, 4.2, 4.3, 4.4, 4.5, 4.6)
    # NOTA: 4.2 demorará muito tempo! (5-10 minutos)
    RUN_META_2_TAREFA_4 = True
    
    # --- FIM PAINEL DE CONTROLO ---

    # --- Execução ---
    
    dados_todos = None # Cache para os dados raw
    dados_transformados = None # Cache para os módulos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def carregar_dados_raw():
        """ Helper para carregar 'dados_todos' apenas 1 vez """
        nonlocal dados_todos
        if dados_todos is None:
            dados_todos = carregar_dados_todos_participantes(base_dir=script_dir)
        return dados_todos
        
    def carregar_dados_transformados():
        """ Helper para calcular 'dados_transformados' apenas 1 vez """
        nonlocal dados_transformados
        if dados_transformados is None:
            dados_raw = carregar_dados_raw()
            if dados_raw.size > 0:
                dados_transformados = get_dados_transformados(dados_raw)
            else:
                dados_transformados = np.array([])
        return dados_transformados

    # --- Execução META 1 ---

    if RUN_TASK_2_TEST:
        print("--- A executar Tarefa 2 (Teste) ---")
        dados_p1 = carregar_dados_participante(1, base_dir=script_dir)
        print(f"Dados do participante 1 carregados. Dimensão: {dados_p1.shape}")

    if RUN_TASK_3_1:
        if carregar_dados_transformados().size > 0:
            plotar_boxplots_3_1(dados_transformados)
    
    if RUN_TASK_3_2:
        if carregar_dados_transformados().size > 0:
            analisar_densidade_iqr_3_2(dados_transformados)
        
    if RUN_TASK_3_4:
        if carregar_dados_transformados().size > 0:
            plotar_outliers_zscore_3_4(dados_transformados)
        
    if RUN_TASK_3_5:
        if carregar_dados_transformados().size > 0:
            comparar_densidades_3_5(dados_transformados)

    if RUN_TASK_3_7:
        if carregar_dados_raw().size > 0:
            analisar_outliers_kmeans_3_7(dados_todos, n_clusters_lista=[2, 3, 5])
            
    if RUN_TASK_3_7_1:
        if carregar_dados_raw().size > 0:
            analisar_outliers_dbscan_3_7_1(dados_todos)
            
    # --- Execução META 2 ---
    
    if RUN_META_2_TAREFA_4:
        
        # Tarefa 4.1 (Usa dados transformados - módulos)
        if carregar_dados_transformados().size > 0:
            analisar_significancia_estatistica_4_1(dados_transformados)
        else:
            print("[ERRO] Não foi possível executar Tarefa 4.1 (sem dados).")
            return
            
        # Tarefa 4.2 (Usa dados raw)
        # Este é o passo mais demorado
        if carregar_dados_raw().size > 0:
            # X_features: (n_janelas, n_features)
            # y_labels: (n_janelas,)
            X_features, y_labels, feature_names = extrair_features_4_2(dados_todos)
        else:
            print("[ERRO] Não foi possível executar Tarefa 4.2 (sem dados).")
            return

        if X_features.size == 0:
            print("\n[ERRO] A Tarefa 4.2 não retornou features.")
            print("Não é possível continuar para 4.3, 4.4, 4.5, 4.6.")
        else:
            # Tarefa 4.3 e 4.4 (PCA)
            analisar_pca_4_3_4_4(X_features)
            
            # Tarefa 4.5 e 4.6 (Seleção de Features)
            analisar_selecao_features_4_5_4_6(X_features, y_labels, feature_names)

    print("\nExecução concluída.")


if __name__ == "__main__":
    main()