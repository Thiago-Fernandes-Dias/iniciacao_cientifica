from lib.repositories.results_repository_factory import results_repository_factory
import numpy as np
import matplotlib.pyplot as plt
from typing import Any


def gerar_histograma_threshold_horizontal(repo: Any, dataset_name: str):
    """
    Carrega os resultados, calcula os thresholds e gera um histograma
    com barras horizontais salvo em PNG.

    Args:
        repo: A factory de repositórios para ler os resultados.
        dataset_name: O nome do dataset (ex: 'CMU' ou 'Keyrecs').
    """
    print(f"Processando dataset: {dataset_name}...")

    # Carrega os dados (mesma lógica de antes)
    st_results_per_user = repo.read_results(
        f"Magalhães com HPO por usuário ({dataset_name})"
    )
    st_results_global = repo.read_results(f"Magalhães com HPO global ({dataset_name})")

    if st_results_global is None or st_results_per_user is None:
        print(f"Resultados não encontrados para o dataset {dataset_name}.")
        return

    # Processamento dos dados (mesma lógica de antes)
    ts_per_user: dict[str, list[float]] = {}
    if st_results_per_user:
        for hp in st_results_per_user.hp_per_seed:
            for user_id, params in hp.items():
                if user_id not in ts_per_user:
                    ts_per_user[user_id] = []
                if "threshold" in params:
                    ts_per_user[user_id].append(params["threshold"])

    avg_t_per_user = {
        user_id: np.mean(thresholds)
        for user_id, thresholds in ts_per_user.items()
        if thresholds
    }

    ts_global_per_seed = [
        hp["threshold"] for hp in st_results_global.hp_per_seed[:5] if "threshold" in hp
    ]

    if not ts_global_per_seed:
        print(f"Nenhum threshold global encontrado para {dataset_name}.")
        return

    ts_global_avg = np.mean(ts_global_per_seed).item()

    # --- Plotagem do Gráfico Horizontal ---
    plt.figure(figsize=(6, 8))  # Ajustado para melhor visualização horizontal

    # 1. Calcular manualmente os dados do histograma
    counts, bin_edges = np.histogram(list(avg_t_per_user.values()), bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Usar plt.barh para plotar as barras horizontais
    plt.barh(
        bin_centers,
        counts,
        height=(bin_edges[1] - bin_edges[0]),
        edgecolor="black",
        alpha=0.75,
    )

    # 3. Mudar a linha vertical para horizontal com axhline
    plt.axhline(
        ts_global_avg,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Threshold global: {ts_global_avg:.2f}",
    )

    # 4. Inverter os rótulos dos eixos
    plt.title(f"Distribuição de Thresholds Médios por Usuário ({dataset_name})")
    plt.ylabel("Threshold médio")
    plt.xlabel("Quantidade de usuários")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.7)  # Grid no eixo x
    plt.tight_layout()

    output_filename = f"histograma_thresholds_{dataset_name.lower()}.png"
    plt.savefig(output_filename)
    print(f"Gráfico salvo como '{output_filename}'")
    plt.close()


# --- Script Principal ---
if __name__ == "__main__":
    try:
        repo = results_repository_factory()

        # Gerar os gráficos para cada dataset
        gerar_histograma_threshold_horizontal(repo, "CMU")
        gerar_histograma_threshold_horizontal(repo, "Keyrecs")

        print("\nProcessamento concluído.")
    except Exception as e:
        if "No module named 'lib'" in str(e):
            print(
                "\nAVISO: O código está correto, mas não pôde ser executado aqui por falta da sua biblioteca customizada ('lib'). Execute-o em seu ambiente de projeto."
            )
        else:
            print(f"Ocorreu um erro inesperado: {e}")
