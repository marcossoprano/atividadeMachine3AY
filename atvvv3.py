from pathlib import Path
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Caminho do arquivo no Colab
DATA_FILE = Path("./barrettII_eyes_clustering.xlsx")

FEATURES = ["AL", "ACD", "WTW", "K1", "K2"]

# Agora o modelo final será ajustado com 5 grupos
FINAL_K = 5


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Arquivo de dados incompleto. Faltam colunas: {missing_cols}")

    return df


def preprocess_features(df: pd.DataFrame):
    X = df[FEATURES].copy()

    if X.isna().any().any():
        print("Dados faltantes detectados nas variáveis de cluster. Aplicando preenchimento por mediana.")
        X = X.fillna(X.median())

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def evaluate_k_values(X_scaled, ks=range(2, 8)):
    records = []

    for k in ks:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50
        )

        labels = model.fit_predict(X_scaled)

        records.append({
            "K": k,
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        })

    return pd.DataFrame(records)


def fit_final_model(X_scaled, n_clusters: int):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=50
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def print_cluster_report(df: pd.DataFrame, labels, n_clusters: int):
    df = df.copy()
    df["Grupo"] = labels

    percentual = df["Grupo"].value_counts(normalize=True).sort_index() * 100
    cluster_summary = df.groupby("Grupo")[FEATURES].agg(["mean", "std", "count"])

    print("\n===== Relatório de Clusters ====")
    print(f"Número total de olhos: {len(df)}")
    print(f"Grupos usados para caracterização: {n_clusters}\n")

    print("Distribuição dos grupos (%):")
    print(percentual.round(1).to_string())

    print("\nPerfil médio por grupo:")
    print(cluster_summary.round(3).to_string())

    if "Correto" in df.columns:
        correct_freq = pd.crosstab(df["Grupo"], df["Correto"], normalize="index") * 100
        print("\nFrequência da coluna Correto por grupo (%):")
        print(correct_freq.round(1).to_string())
    else:
        print("\nA coluna 'Correto' não foi encontrada no arquivo.")

    print("\nInterpretação dos perfis de olhos baseada em AL, ACD, WTW, K1 e K2:")
    for group in sorted(df["Grupo"].unique()):
        mean_values = df.loc[df["Grupo"] == group, FEATURES].mean()
        print(
            f"Grupo {group}: "
            f"AL={mean_values['AL']:.2f}, "
            f"ACD={mean_values['ACD']:.2f}, "
            f"WTW={mean_values['WTW']:.2f}, "
            f"K1={mean_values['K1']:.2f}, "
            f"K2={mean_values['K2']:.2f}"
        )


def main():
    print(f"Carregando dados de {DATA_FILE}")

    df = load_data(DATA_FILE)

    print("\n===== Estatísticas Descritivas por Feature =====")
    for feature in FEATURES:
        min_val = df[feature].min()
        max_val = df[feature].max()
        median_val = df[feature].median()
        mean_val = df[feature].mean()
        print(f"{feature}: Min={min_val:.2f}, Max={max_val:.2f}, Median={median_val:.2f}, Mean={mean_val:.2f}")
    print("=================================================\n")

    X_scaled, scaler = preprocess_features(df)

    scores = evaluate_k_values(X_scaled)

    print("Avaliação de K entre 2 e 7:")
    print(scores.to_string(index=False, float_format="%.4f"))

    print(f"\nTestando o modelo final com K={FINAL_K}.")
    model, labels = fit_final_model(X_scaled, FINAL_K)

    print_cluster_report(df, labels, FINAL_K)


if __name__ == "__main__":
    main()