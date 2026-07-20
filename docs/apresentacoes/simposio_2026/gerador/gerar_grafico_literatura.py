from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    output_path = Path(__file__).with_name("grafico_turboquant_literatura.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=180)
    fig.patch.set_facecolor("white")

    memory_labels = ["Baseline\n32-bit", "TurboQuant\n3-bit"]
    memory_values = [1.0, 1.0 / 6.0]
    memory_colors = ["#9AA9B5", "#2D8A5E"]

    axes[0].bar(memory_labels, memory_values, color=memory_colors, width=0.58)
    axes[0].set_title("Memória relativa do KV cache", fontsize=13, weight="bold")
    axes[0].set_ylabel("Escala normalizada", fontsize=11)
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].text(1, memory_values[1] + 0.05, ">= 6x menor", ha="center", va="bottom", fontsize=11, color="#1E5F42", weight="bold")
    axes[0].text(0, memory_values[0] + 0.03, "1.0x", ha="center", va="bottom", fontsize=10, color="#4D5A66")
    axes[0].text(1, memory_values[1] - 0.02, "0.17x", ha="center", va="top", fontsize=10, color="#1E5F42")

    speed_labels = ["Baseline\n32-bit", "TurboQuant\n4-bit"]
    speed_values = [1.0, 8.0]
    speed_colors = ["#9AA9B5", "#1F5A95"]

    axes[1].bar(speed_labels, speed_values, color=speed_colors, width=0.58)
    axes[1].set_title("Velocidade relativa em attention logits", fontsize=13, weight="bold")
    axes[1].set_ylabel("Speedup relativo", fontsize=11)
    axes[1].set_ylim(0, 8.9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].text(1, speed_values[1] + 0.18, "até 8x", ha="center", va="bottom", fontsize=11, color="#1F5A95", weight="bold")
    axes[1].text(0, speed_values[0] + 0.12, "1.0x", ha="center", va="bottom", fontsize=10, color="#4D5A66")
    axes[1].text(1, speed_values[1] - 0.22, "8.0x", ha="center", va="top", fontsize=10, color="white", weight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#B8C1C8")
        axis.spines["bottom"].set_color("#B8C1C8")
        axis.tick_params(labelsize=10)

    fig.suptitle(
        "Evidências iniciais da literatura a favor da quantização",
        fontsize=16,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Dados adaptados do Google Research Blog (24/03/2026). Valores resumem ordens de grandeza reportadas e não representam resultados deste trabalho.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    plt.tight_layout(rect=(0.03, 0.08, 0.98, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Chart written to: {output_path}")


if __name__ == "__main__":
    main()
