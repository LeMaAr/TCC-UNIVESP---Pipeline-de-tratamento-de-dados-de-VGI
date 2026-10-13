"""
Cálculo de Métricas Espaciais Robustas: Mediana Geométrica vs Média Simples
"""

import os
import numpy as np
import pandas as pd


def calcular_mediana_geometrica(pontos: np.ndarray, tolerancia: float = 1e-6) -> np.ndarray:
    """
    Calcula a Mediana Geométrica (ponto que minimiza a soma das distâncias euclidianas)
    utilizando o algoritmo iterativo de Weiszfeld.
    """
    y = np.mean(pontos, axis=0)
    while True:
        distancias = np.linalg.norm(pontos - y, axis=1)
        nao_nulos = distancias != 0
        if not np.any(nao_nulos):
            return y
        pesos = 1.0 / distancias[nao_nulos]
        pesos_norm = pesos / np.sum(pesos)
        t = np.sum(pontos[nao_nulos] * pesos_norm[:, np.newaxis], axis=0)

        num_zeros = len(pontos) - np.sum(nao_nulos)
        if num_zeros == 0:
            y_novo = t
        else:
            r = np.linalg.norm((t - y) * np.sum(pesos))
            gamma = min(1.0, 0.0 if r == 0 else num_zeros / r)
            y_novo = (1 - gamma) * t + gamma * y

        if np.linalg.norm(y - y_novo) < tolerancia:
            return y_novo
        y = y_novo


def comparar_centroides(
    df: pd.DataFrame,
    coluna_especie: str = "vernacularName",
    dir_saida: str = "data/processed"
) -> pd.DataFrame:
    """
    Gera tabela comparativa de centroides para as espécies com maior amostragem
    e exporta para CSV no diretório indicado.
    """
    os.makedirs(dir_saida, exist_ok=True)

    resultados = []
    top_especies = df[coluna_especie].value_counts().head(5).index

    for especie in top_especies:
        sub = df[df[coluna_especie] == especie]
        pontos = sub[["decimalLatitude", "decimalLongitude"]].values

        media = np.mean(pontos, axis=0)
        mediana = calcular_mediana_geometrica(pontos)

        # Distância aproximada em metros entre a média e a mediana (aprox: 111.139m por grau)
        deslocamento_metros = np.linalg.norm(media - mediana) * 111139

        resultados.append({
            "Espécie": especie,
            "N_Registros": len(sub),
            "Latitude_Média": round(media[0], 6),
            "Longitude_Média": round(media[1], 6),
            "Latitude_Mediana": round(mediana[0], 6),
            "Longitude_Mediana": round(mediana[1], 6),
            "Deslocamento_Estimado_Metros": round(deslocamento_metros, 2)
        })

    centroides = pd.DataFrame(resultados)

    caminho_csv = os.path.join(dir_saida, "comparacao_centroides.csv")
    centroides.to_csv(caminho_csv, index=False)
    print(f"\n[OK] Arquivo gerado com sucesso em: {os.path.abspath(caminho_csv)}")

    return centroides


if __name__ == "__main__":
    # Resolve caminhos relativos independente de onde o terminal estiver aberto
    caminhos_possiveis = [
        "data/processed/dados_validados_dwc.csv",
        "../data/processed/dados_validados_dwc.csv"
    ]

    caminho_encontrado = None
    for p in caminhos_possiveis:
        if os.path.exists(p):
            caminho_encontrado = p
            break

    if caminho_encontrado:
        # Define diretório de saída coerente com o local de entrada
        diretorio_saida = os.path.dirname(caminho_encontrado)
        df = pd.read_csv(caminho_encontrado)
        tabela_comparativa = comparar_centroides(df, dir_saida=diretorio_saida)
        print("\n=== COMPARAÇÃO DE CENTROIDES (TOP ESPÉCIES) ===")
        print(tabela_comparativa.to_string(index=False))
    else:
        print("[ERRO] Arquivo 'dados_validados_dwc.csv' não foi encontrado!")
        print("Certifique-se de executar primeiro o 'pipeline.py' para gerar a base tratada.")