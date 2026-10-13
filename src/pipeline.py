"""
Pipeline de Qualificação e Limpeza de Dados Espaciais de VGI
Estudo de caso: Registros Ornitológicos da plataforma TocaBirds em Ilhabela
TCC - Bacharelado em Ciência de Dados - UNIVESP
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def aplicar_clusterizacao_dbscan(
    df: pd.DataFrame,
    raio_metros: float = 800.0,
    min_amostras: int = 10
) -> pd.DataFrame:
    """
    Aplica o algoritmo de vizinhança DBSCAN com métrica Haversine
    para detectar clusters de alta densidade e isolar ruídos espaciais.
    """
    # Conversão das coordenadas para radianos para cálculo geodésico exato
    coords_rad = np.radians(df[["decimalLatitude", "decimalLongitude"]].values)

    # Epsilon em radianos: raio em km dividido pelo raio médio da Terra (6.371 km)
    raio_km = raio_metros / 1000.0
    epsilon_rad = raio_km / 6371.0

    db = DBSCAN(eps=epsilon_rad, min_samples=min_amostras, metric="haversine")
    clusters = db.fit_predict(coords_rad)

    df["clusterID"] = clusters
    # clusterID == -1 representa observações esparsas/ruído no DBSCAN
    df["isHighDensityCluster"] = df["clusterID"] != -1

    # Mapeamento do protocolo Darwin Core baseado na densidade encontrada pelo algoritmo
    df["samplingProtocol"] = np.where(
        df["isHighDensityCluster"],
        "StationaryCount",
        "IncidentalObservation"
    )

    # Atribuição semântica contextual baseada nos clusters encontrados
    df["locality"] = np.where(
        df["clusterID"] == 0,
        "Fazenda da Toca (Cluster 0)",
        np.where(
            df["clusterID"] == 1,
            "Região Central / Balsa (Cluster 1)",
            "Ilhabela (Observação Esparsa)"
        )
    )

    return df


def sanitizar_dados_vgi(caminho_entrada: str, dir_saida: str = "data/processed"):
    """
    Executa a auditoria, limpeza, enriquecimento espacial e transformação da base bruta.
    """
    os.makedirs(dir_saida, exist_ok=True)

    # 1. Ingestão da base bruta
    df_raw = pd.read_csv(caminho_entrada)
    total_bruto = len(df_raw)

    # 2. Expurgar registros de logs de aplicação (eventos de sistema gravados como avistamentos)
    mascara_logs = df_raw["bird_name"].str.contains("PDF Generated", case=False, na=False)
    df_sem_logs = df_raw[~mascara_logs].copy()
    logs_removidos = total_bruto - len(df_sem_logs)

    # 3. Segregação de registros sem coordenadas para quarentena
    mascara_nulos = df_sem_logs["latitude"].isna() | df_sem_logs["longitude"].isna()
    df_quarentena = df_sem_logs[mascara_nulos].copy()
    df_com_coords = df_sem_logs[~mascara_nulos].copy()

    # 4. Delimitação Geográfica (Delimitação territorial do município de Ilhabela)
    LAT_MIN, LAT_MAX = -24.10, -23.60
    LON_MIN, LON_MAX = -45.60, -45.15

    mascara_dentro_ilhabela = (
        (df_com_coords["latitude"] >= LAT_MIN) &
        (df_com_coords["latitude"] <= LAT_MAX) &
        (df_com_coords["longitude"] >= LON_MIN) &
        (df_com_coords["longitude"] <= LON_MAX)
    )

    df_validados = df_com_coords[mascara_dentro_ilhabela].copy()
    df_outliers_geo = df_com_coords[~mascara_dentro_ilhabela].copy()

    # 5. Aplicação do padrão Darwin Core (DwC)
    df_dwc = pd.DataFrame()
    df_dwc["occurrenceID"] = df_validados["id"]
    df_dwc["vernacularName"] = df_validados["bird_name"].str.strip()
    df_dwc["eventDate"] = pd.to_datetime(df_validados["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df_dwc["decimalLatitude"] = df_validados["latitude"].round(6)
    df_dwc["decimalLongitude"] = df_validados["longitude"].round(6)
    df_dwc["geodeticDatum"] = "WGS84"
    df_dwc["basisOfRecord"] = "HumanObservation"
    df_dwc["countryCode"] = "BR"
    df_dwc["stateProvince"] = "São Paulo"
    df_dwc["municipality"] = "Ilhabela"

    # 6. Clusterização Espacial e Detecção de Viés via DBSCAN
    df_dwc = aplicar_clusterizacao_dbscan(
        df_dwc,
        raio_metros=800.0,
        min_amostras=10
    )

    # 7. Exportação dos artefatos higienizados
    caminho_dwc = os.path.join(dir_saida, "dados_validados_dwc.csv")
    caminho_quarentena = os.path.join(dir_saida, "quarentena_nao_espacial.csv")
    caminho_outliers = os.path.join(dir_saida, "outliers_geograficos.csv")

    df_dwc.to_csv(caminho_dwc, index=False)
    df_quarentena.to_csv(caminho_quarentena, index=False)
    df_outliers_geo.to_csv(caminho_outliers, index=False)

    # 8. Relatório Consolidado de Execução
    metricas = {
        "Total de Registros Brutos": total_bruto,
        "Logs de Aplicação Expurgados": logs_removidos,
        "Avistamentos Genuínos": len(df_sem_logs),
        "Registros em Quarentena (Sem Coordenadas)": len(df_quarentena),
        "Registros com Coordenadas": len(df_com_coords),
        "Outliers Fora de Ilhabela (Portugal, SP, etc.)": len(df_outliers_geo),
        "Registros Espacialmente Validados em Ilhabela": len(df_dwc),
        " - Cluster 0 - Fazenda da Toca (StationaryCount)": int((df_dwc["clusterID"] == 0).sum()),
        " - Cluster 1 - Região Central/Balsa (StationaryCount)": int((df_dwc["clusterID"] == 1).sum()),
        " - Observações Esparsas / Ruído (IncidentalObservation)": int((df_dwc["clusterID"] == -1).sum()),
        "Espécies Distintas Validadas": df_dwc["vernacularName"].nunique(),
    }

    print("\n=== RELATÓRIO DO PIPELINE DE QUALIFICAÇÃO ===")
    for metrica, valor in metricas.items():
        print(f"{metrica}: {valor}")
    print("============================================\n")

    return df_dwc, df_quarentena, metricas


if __name__ == "__main__":
    caminhos_possiveis = [
        "data/raw/avistamentos-producao-2026-09-27_16-05.csv",
        "avistamentos-producao-2026-09-27_16-05.csv",
        "../data/raw/avistamentos-producao-2026-09-27_16-05.csv"
    ]

    caminho_encontrado = None
    for p in caminhos_possiveis:
        if os.path.exists(p):
            caminho_encontrado = p
            break

    if caminho_encontrado:
        dir_base = os.path.dirname(caminho_encontrado)
        dir_saida = "data/processed" if "data" in dir_base or dir_base == "" else os.path.join(dir_base, "../processed")
        sanitizar_dados_vgi(caminho_encontrado, dir_saida=dir_saida)
    else:
        print("[ERRO] Arquivo de dados bruto não encontrado!")