# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ==========================
# 1) Cấu hình chung cho biểu đồ
# ==========================
plt.rcParams.update({
    "font.family": ["Arial", "Calibri", "DejaVu Sans"],
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

sns.set_theme(style="whitegrid", context="talk")

# Palette đồng nhất (xanh dương pastel/chuyên nghiệp)
PALETTE = {
    "primary": "#4C78A8",
    "secondary": "#5DA5DA",
    "accent": "#9DC3E6",
    "soft": "#EAF2FF",
    "danger": "#F2A7A0",
    "text": "#1F2937",
    "grid": "#D9E2EC",
    "bg": "#F7FAFC"
}

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================
# 2) Biểu đồ ①: Thời gian xử lý từng bước
# (Horizontal Bar Chart)
# ==========================
def plot_processing_time():
    steps = [
        "Mesh Reconstruction",
        "Depth Estimation",
        "UV Mapping",
        "Point Cloud",
        "Background Removal"
    ]
    times = [34, 21, 16, 12, 5]  # giả định

    df = pd.DataFrame({
        "Bước": steps,
        "Thời gian (s)": times
    })

    # Sắp xếp giảm dần để thanh từ trên xuống dưới đúng thứ tự thời gian
    df = df.sort_values("Thời gian (s)", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(
        df["Bước"],
        df["Thời gian (s)"],
        color=PALETTE["primary"],
        edgecolor="white",
        linewidth=1.2,
        height=0.7
    )

    # Đảo ngược trục Y để thanh dài nhất nằm ở trên
    ax.invert_yaxis()

    ax.set_title("Thời gian xử lý từng bước", pad=10)
    ax.set_xlabel("Thời gian xử lý (giây)")
    ax.set_ylabel("Các bước xử lý")

    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35, color=PALETTE["grid"])
    ax.set_axisbelow(True)

    # Ghi giá trị số lên đầu mỗi thanh
    for bar, val in zip(bars, df["Thời gian (s)"]):
        ax.text(
            val + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{val}s",
            va="center",
            ha="left",
            fontsize=10,
            color=PALETTE["text"],
            fontweight="bold",
        )

    # Tùy chỉnh viền
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["grid"])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_thoi_gian_xu_ly.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================
# 3) Biểu đồ ②: Tỷ lệ thành công của Pipeline
# (Donut Chart)
# ==========================
def plot_pipeline_success():
    labels = ["Thành công", "Lỗi"]
    sizes = [188, 4]
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["primary"], PALETTE["danger"]]

    def autopct_func(pct):
        val = int(round(pct * total / 100.0))
        return f"{val}\n{pct:.1f}%"

    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        autopct=autopct_func,
        pctdistance=0.62,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.2),
        textprops=dict(color=PALETTE["text"], fontsize=11)
    )

    # Làm nổi bật phần "Thành công"
    wedges[0].set_edgecolor(PALETTE["primary"])
    wedges[0].set_linewidth(2.0)

    # Vòng tròn ở giữa
    centre_circle = plt.Circle((0, 0), 0.55, fc=PALETTE["bg"])
    ax.add_artist(centre_circle)

    # Text trung tâm
    ax.text(
        0, 0,
        f"Thành công\n{sizes[0]}/{total}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    ax.set_title("Tỷ lệ thành công của Pipeline", pad=20)

    # Legend ở bên phải
    legend_labels = [f"{lab}: {val} ({val/total*100:.1f}%)" for lab, val in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=10
    )

    ax.axis("equal")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_ty_le_thanh_cong.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================
# 4) Biểu đồ ③: Đánh giá chất lượng từng bước
# (Heatmap Table)
# ==========================
def plot_quality_heatmap():
    steps = ["Tách nền", "Crop", "Depth", "Mesh", "Texture"]
    scores = [5, 5, 4, 4, 5]       # thang điểm 5
    labels = ["Tốt", "Tốt", "Khá", "Khá", "Tốt"]

    # Tạo DataFrame 1 hàng, 5 cột để làm heatmap dạng bảng
    df = pd.DataFrame([scores], index=["Đánh giá"], columns=steps)

    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=180)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    annot = np.array([labels], dtype=object)

    sns.heatmap(
        df,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=4,
        vmax=5,
        linewidths=0.8,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Điểm số (thang 5)", "shrink": 0.9},
        annot_kws={"fontsize": 10, "fontweight": "bold", "color": PALETTE["text"]},
        ax=ax
    )

    ax.set_title("Đánh giá chất lượng từng bước")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_heatmap_chat_luong.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================
# 5) Biểu đồ ④: Workflow kết quả
# (Sơ đồ luồng công việc - fallback về matplotlib nếu graphviz không sẵn)
# ==========================
def plot_workflow():
    nodes = ["Ảnh", "Mask", "Crop", "Depth", "Mesh", "Texture"]

    # Cố gắng dùng graphviz nếu có sẵn
    try:
        from graphviz import Digraph

        g = Digraph("workflow", format="png")
        g.attr(rankdir="LR", bgcolor="white", splines="ortho", nodesep="0.55", ranksep="0.75")
        g.attr("node", shape="box", style="rounded,filled", fillcolor=PALETTE["soft"],
               color=PALETTE["primary"], penwidth="1.2", fontname="Arial", fontsize="11")
        g.attr("edge", color=PALETTE["primary"], penwidth="1.4", arrowsize="0.8")

        for n in nodes:
            g.node(n, n)

        for i in range(len(nodes) - 1):
            g.edge(nodes[i], nodes[i + 1])

        g.render(filename=str(OUTPUT_DIR / "04_workflow"), cleanup=True)
        return

    except Exception:
        # Fallback: vẽ bằng matplotlib
        fig, ax = plt.subplots(figsize=(12, 2.6), dpi=180)
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        x_positions = np.linspace(0.10, 0.90, len(nodes))
        box_w, box_h = 0.10, 0.32

        for x, node in zip(x_positions, nodes):
            box = FancyBboxPatch(
                (x - box_w / 2, 0.35),
                box_w,
                box_h,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                linewidth=1.2,
                edgecolor=PALETTE["primary"],
                facecolor=PALETTE["soft"]
            )
            ax.add_patch(box)
            ax.text(x, 0.51, node, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=PALETTE["text"])

            if node != nodes[-1]:
                next_x = x + (x_positions[1] - x_positions[0])
                arrow = FancyArrowPatch(
                    (x + box_w / 2, 0.51),
                    (next_x - box_w / 2, 0.51),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color=PALETTE["primary"]
                )
                ax.add_patch(arrow)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.2, 0.8)
        ax.axis("off")
        ax.set_title("Workflow kết quả từ ảnh đến mô hình 3D", pad=12)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "04_workflow.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# ==========================
# 6) Chạy toàn bộ
# ==========================
if __name__ == "__main__":
    plot_processing_time()
    plot_pipeline_success()
    plot_quality_heatmap()
    plot_workflow()

    print("Đã tạo xong các file hình ảnh trong thư mục:", OUTPUT_DIR.resolve())