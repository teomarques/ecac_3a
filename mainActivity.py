# -*- coding: utf-8 -*-
"""
EA/ECAC 2025 - Trabalho Prático 1
mainActivity.py

Implementação das tarefas da Meta 1:
1.  Carregamento de dados
2.  Análise de Outliers (IQR, Z-Score, K-Means)
"""
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessário para plots 3D [cite: 81]
from sklearn.cluster import DBSCAN      # Apenas para o bónus 3.7.1 [cite: 82]

# --- Constantes Globais (Baseado no PDF) ---
# Mapeamento 1-para-1 com o PDF (1-based-indexing) para 0-based-indexing
COL_DEVICE_ID = 0  # Coluna 1 [cite: 29]
COL_ACC_X = 1      # Coluna 2 [cite: 30]
COL_ACC_Y = 2      # Coluna 3 [cite: 31]
COL_ACC_Z = 3      # Coluna 4 [cite: 32]
COL_GYRO_X = 4     # Coluna 5 [cite: 33]
COL_GYRO_Y = 5     # Coluna 6 [cite: 34]
COL_GYRO_Z = 6     # Coluna 7 [cite: 35]
COL_MAG_X = 7      # Coluna 8 [cite: 37]
COL_MAG_Y = 8      # Coluna 9 [cite: 38]
COL_MAG_Z = 9      # Coluna 10 [cite: 39]
COL_TIMESTAMP = 10 # Coluna 11 [cite: 40]
COL_ACTIVITY = 11  # Coluna 12 [cite: 41]

# Grupos de colunas para cálculo dos módulos
MODULO_INDICES = [
    (COL_ACC_X, COL_ACC_Y, COL_ACC_Z),  # Aceleração [cite: 54]
    (COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z), # Giroscópio [cite: 54]
    (COL_MAG_X, COL_MAG_Y, COL_MAG_Z)  # Magnetómetro [cite: 54]
]

# Labels para gráficos
VAR_LABELS = [
    "Módulo Acelerômetro",
    "Módulo Giroscópio",
    "Módulo Magnetômetro"
]
SENSOR_LABELS = {
    1: "Pulso Esquerdo",  # ID 1 [cite: 44]
    2: "Pulso Direito",   # ID 2 [cite: 44]
    3: "Peito",           # ID 3 [cite: 44]
    4: "Perna Sup. Direita", # ID 4 [cite: 44]
    5: "Perna Inf. Esquerda" # ID 5 [cite: 44]
}

# --- Tarefa 2: Carregamento de Dados [cite: 51] ---

def carregar_dados_participante(num_participante, base_dir="."):
    """
    (Tarefa 2) Lê todos os 5 ficheiros CSV do participante e devolve um array NumPy.
    """
    part_dir = os.path.join(base_dir, f"part{num_participante}")
    dados = []

    for i in range(1, 6): # Dispositivos 1 a 5 [cite: 18, 20, 21, 22, 23]
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
    [cite: 17, 58]
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

# --- Tarefa 3.1: Boxplot de Atividades [cite: 58] ---

def plotar_boxplots_3_1(dados_transformados):
    """
    (Tarefa 3.1) Gera 15 boxplots (3 vars x 5 sensores).
    Eixo X: Atividade (1-16) [cite: 42, 58]
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
            for a in range(1, 17): # Atividades 1 a 16 [cite: 42]
                activity_mask = dados_sensor[:, 1] == a
                dados_ativ = dados_sensor[activity_mask, var_idx + 2] # +2 para aceder às cols de módulo
                dados_box.append(dados_ativ)
                
            # 3. Plotar
            # O `labels=` define os ticks do eixo X
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
    # plt.show() # Descomentar para mostrar interativamente

# --- Tarefa 3.2: Análise Densidade de Outliers (IQR) [cite: 61] ---

def analisar_densidade_iqr_3_2(dados_transformados):
    """
    (Tarefa 3.2) Analisa e comenta a densidade de outliers.
    Usa método IQR (Tukey)[cite: 62].
    Filtra apenas para o Pulso Direito (ID 2)[cite: 62, 44].
    """
    print("\n--- Início Tarefa 3.2: Densidade de Outliers (IQR - Pulso Direito) ---")
    
    sensor_id_foco = 2 # Pulso Direito [cite: 62]
    
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
            
            n_r = len(amostras) # nr = número total de pontos [cite: 68]
            if n_r == 0:
                continue # Pula atividades sem dados

            # 3. Aplicar método IQR (Tukey) [cite: 62]
            q1 = np.percentile(amostras, 25)
            q3 = np.percentile(amostras, 75)
            iqr = q3 - q1
            
            limite_inf = q1 - (1.5 * iqr)
            limite_sup = q3 + (1.5 * iqr)
            
            # 4. Contar outliers (no)
            outliers_mask = (amostras < limite_inf) | (amostras > limite_sup)
            n_o = np.sum(outliers_mask) # no = número de outliers [cite: 68]
            
            # 5. Calcular densidade (d) [cite: 64]
            densidade = (n_o / n_r) * 100
            
            print(f"| {var_label:21} | {a:9} | {n_r:12} | {n_o:15} | {densidade:13.2f}% |")

    print("\n--- Fim Tarefa 3.2 ---")

# --- Tarefa 3.3: Rotina Z-Score [cite: 69] ---

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

# --- Tarefa 3.4: Plots Z-Score [cite: 71] ---

def plotar_outliers_zscore_3_4(dados_transformados):
    """
    (Tarefa 3.4) Gera plots análogos a 3.1, mas usando Z-Score.
    Outliers a vermelho, inliers a azul[cite: 72].
    Testa k = 3, 3.5 e 4[cite: 73].
    """
    print("A gerar gráficos para Tarefa 3.4...")
    
    k_valores = [3, 3.5, 4]  # [cite: 73]
    
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
        plt.savefig(f"meta1_tarefa_3_4_zscore_k{k}.png")
        print(f"Gráfico 'meta1_tarefa_3_4_zscore_k{k}.png' guardado.")
        # plt.show()

# --- Tarefa 3.5: Comparação IQR vs Z-Score [cite: 76] ---

def comparar_densidades_3_5(dados_transformados):
    """
    (Tarefa 3.5) Compara e discute resultados de 3.1 (IQR) e 3.4 (Z-Score).
    Foco: Apenas sensores do pulso direito (ID 2)[cite: 76].
    """
    print("\n--- Início Tarefa 3.5: Comparação Densidades (IQR vs Z-Score) - Pulso Direito ---")
    
    sensor_id_foco = 2 # Pulso Direito [cite: 76]
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

    print("\n--- Discussão (Técnica de Feynman) ---")
    print("O que esta tabela nos diz (Evocação Ativa):")
    print("1. O que é o Método IQR? (3.1/3.2) [cite: 62]")
    print("   - O 'método de Tukey' usado nos boxplots define outliers como pontos 1.5x 'tamanho da caixa' (IQR) acima ou abaixo da caixa.")
    print("   - Vantagem: É 'não-paramétrico'. Não assume que os dados são uma 'curva de sino' (distribuição normal). É robusto a dados assimétricos.")
    print("\n2. O que é o Método Z-Score? (3.3/3.4) [cite: 69, 71]")
    print("   - Define outliers como pontos que estão a 'k' desvios-padrão da média.")
    print("   - Vantagem: É bom para dados com distribuição normal.")
    print("   - Desvantagem: A própria média e o desvio-padrão são *sensíveis* a outliers. Um outlier muito extremo 'puxa' a média e 'inflaciona' o desvio, fazendo com que o Z-Score *mascare* esse mesmo outlier.")
    print("\n3. Comparação (3.5)[cite: 76]:")
    print("   - Nos dados deste projeto, o Z-Score (k=3) é geralmente *mais restritivo* (encontra menos outliers) do que o IQR.")
    print("   - Porquê? Provavelmente porque as nossas distribuições de dados (ver boxplots 3.1) não são perfeitamente 'normais'. Têm 'caudas longas'.")
    print("   - À medida que 'k' aumenta (3 -> 3.5 -> 4)[cite: 73], a densidade de outliers Z-Score *diminui*, porque a 'barra' para ser considerado um outlier fica mais alta (mais longe da média).")
    print("--- Fim Tarefa 3.5 ---")

# --- Tarefa 3.6: Rotina K-Means [cite: 77] ---

def kmeans_3_6(X, n_clusters, max_iter=100, tol=1e-4):
    """
    (Tarefa 3.6) Implementa o algoritmo k-means.
    """
    # 1. Inicialização: Escolhe n_clusters pontos aleatórios dos dados como centroides
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
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

# --- Tarefa 3.7: Outliers com K-Means e Plot 3D [cite: 79] ---

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
    atividade_foco = 4 # Walk [cite: 42]
    
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
            centroides, labels = kmeans_3_6(X, n_clusters)
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

        # 4. Plot 3D (clusters coloridos distintamente) [cite: 81]
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
        plt.savefig(f"meta1_tarefa_3_7_kmeans_k{n_clusters}.png")
        print(f"Gráfico 'meta1_tarefa_3_7_kmeans_k{n_clusters}.png' guardado.")

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
        plt.savefig(f"meta1_tarefa_3_7_kmeans_k{n_clusters}_sem_outliers.png")
        print(f"Gráfico 'meta1_tarefa_3_7_kmeans_k{n_clusters}_sem_outliers.png' guardado.")
        # plt.show()
        
    print("\n--- Discussão (Comparação 3.7 vs 3.4) ---")
    print("O que este plot 3D nos diz (Evocação Ativa):")
    print("1. O K-Means (3.7) [cite: 79] é 'multivariado'. Encontra outliers no espaço 3D (x,y,z).")
    print("2. O Z-Score (3.4) [cite: 71] foi 'univariado'. Analisou o *módulo* (um único número) de cada vez.")
    print("3. Diferença Chave: Um ponto pode ser um outlier 3D (K-Means) sem ser um outlier 1D (Z-Score).")
    print("   Exemplo: Um valor de (x=1, y=1, z=1) pode ser normal. (x=5, y=1, z=1) pode ser normal. Mas (x=5, y=5, z=5) pode ser uma combinação *impossível* para o sensor, mesmo que 5 não seja um outlier no Z-Score. O K-Means deteta isto.")
    print("--- Fim Tarefa 3.7 ---")

# --- Tarefa 3.7.1 (Bónus): Outliers com DBSCAN [cite: 82] ---

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

    # 1. Correr DBSCAN
    #    Estes parâmetros (eps, min_samples) são a 'alma' do DBSCAN
    #    e precisam de ser afinados.
    db = DBSCAN(eps=0.5, min_samples=10).fit(X)
    
    # 2. Outliers são pontos com label = -1
    outlier_mask = db.labels_ == -1
    inlier_mask = ~outlier_mask
    
    n_o = np.sum(outlier_mask)
    print(f"  DBSCAN (eps=0.5, min=10) encontrou {n_o} outliers ({n_o/X.shape[0]*100:.2f}%)")

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
    plt.savefig("meta1_tarefa_3_7_1_dbscan.png")
    print("Gráfico 'meta1_tarefa_3_7_1_dbscan.png' guardado.")

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
    plt.savefig("meta1_tarefa_3_7_1_dbscan_sem_outliers.png")
    print("Gráfico 'meta1_tarefa_3_7_1_dbscan_sem_outliers.png' guardado.")
    # plt.show()
    print("--- Fim Tarefa 3.7.1 ---")


# --- Função Principal (main)  ---
def main():
    """
    Função principal que orquestra a execução das tarefas da Meta 1.
    """
    
    # --- PAINEL DE CONTROLO ---
    # Defina como True/False para executar/saltar cada tarefa
    
    # Tarefa 2: Carregar dados de 1 participante (teste rápido)
    RUN_TASK_2_TEST = False 
    
    # Tarefa 3.1: Gerar os 15 boxplots (Requer 'dados_todos')
    RUN_TASK_3_1 = True
    
    # Tarefa 3.2: Calcular densidade IQR (Pulso Direito)
    RUN_TASK_3_2 = True
    
    # Tarefa 3.4: Gerar plots Z-Score (Demorado: 3x15 plots)
    RUN_TASK_3_4 = True
    
    # Tarefa 3.5: Gerar tabela comparativa de densidades
    RUN_TASK_3_5 = True
    
    # Tarefa 3.7: Gerar plots K-Means 3D (Exemplo focado)
    RUN_TASK_3_7 = True
    
    # Tarefa 3.7.1 (Bónus): Gerar plot DBSCAN 3D
    RUN_TASK_3_7_1 = True
    
    # --- FIM PAINEL DE CONTROLO ---

    # --- Execução ---

    if RUN_TASK_2_TEST:
        print("--- A executar Tarefa 2 (Teste) ---")
        # Testar a Tarefa 2 [cite: 51]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dados_p1 = carregar_dados_participante(1, base_dir=script_dir)
        print(f"Dados do participante 1 carregados. Dimensão: {dados_p1.shape}")
        if dados_p1.size == 0:
            print("[ERRO] Teste da Tarefa 2 falhou. Verifique o caminho.")
            return # Sai se não conseguir carregar dados de teste
            
    # Para as tarefas 3.1 em diante, precisamos de *todos* os dados [cite: 58]
    # Usamos estas flags para só carregar os dados se for necessário
    if any([RUN_TASK_3_1, RUN_TASK_3_2, RUN_TASK_3_4, RUN_TASK_3_5, RUN_TASK_3_7, RUN_TASK_3_7_1]):
        
        # --- Passo 1: Carregar Dados (Raw) ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dados_todos = carregar_dados_todos_participantes(base_dir=script_dir)
        if dados_todos.size == 0:
            return # Sai se não houver dados

        # --- Passo 2: Transformar Dados (Módulos) [cite: 54] ---
        # (Apenas necessário para 3.1, 3.2, 3.4, 3.5)
        if any([RUN_TASK_3_1, RUN_TASK_3_2, RUN_TASK_3_4, RUN_TASK_3_5]):
            dados_transformados = get_dados_transformados(dados_todos)
        
            if RUN_TASK_3_1:
                plotar_boxplots_3_1(dados_transformados)
            
            if RUN_TASK_3_2:
                analisar_densidade_iqr_3_2(dados_transformados)
                
            if RUN_TASK_3_4:
                plotar_outliers_zscore_3_4(dados_transformados)
                
            if RUN_TASK_3_5:
                comparar_densidades_3_5(dados_transformados)

        # --- Passo 3: Análise Multivariada (Usa dados 'raw') ---
        # (Tarefas 3.7 e 3.7.1 usam os dados 'raw' (x,y,z))
        
        if RUN_TASK_3_7:
            # Testa com 2, 3 e 5 clusters [cite: 80]
            analisar_outliers_kmeans_3_7(dados_todos, n_clusters_lista=[2, 3, 5])
            
        if RUN_TASK_3_7_1:
            analisar_outliers_dbscan_3_7_1(dados_todos)
            
    print("\nExecução da Meta 1 concluída.")


if __name__ == "__main__":
    main()