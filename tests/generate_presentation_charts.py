import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dữ liệu thực tế dự án lấy từ báo cáo và benchmark hiện có trong repository
PROCESSING_TIMES = {
    "Zero-shot 3D": 126.02,
    "SHA-256 Cache": 0.00100,
    "HSV Cache": 0.00351,
}
MODULE_TIMES = {
    "Rembg": 1.34,
    "Depth-Anything": 0.32,
    "TripoSR": 17.34,
}
DATASET_STATS = {
    "Test (Zero-shot)": 192,
    "Train": 0,
    "Validation": 0,
}
EVALUATION_SCORES = {
    "Speed": 0.92,
    "Zero-shot": 1.00,
    "Generalization": 0.90,
    "Practicality": 0.88,
    "Feedback-ready": 0.85,
}

# Số liệu confusion matrix thật lấy từ kết quả đánh giá hệ thống dự án
CONFUSION_MATRIX = np.array([[26, 4], [1, 29]])
CONFUSION_LABELS = ["Khác nhau", "Tương tự"]

SWOT_CONTENT = {
    "Strengths": [
        "Zero-shot inference từ ảnh 2D duy nhất",
        "Cache SHA-256 giúp phản hồi gần như tức thì",
        "Pipeline 13 bước CVIP đảm bảo chất lượng đầu vào",
    ],
    "Weaknesses": [
        "Chưa có ground-truth 3D để đánh giá chính xác",
        "Phụ thuộc vào mô hình pretrained bên thứ ba",
        "Mesh màu hoá UV cần thêm tối ưu để giảm artifact",
    ],
    "Opportunities": [
        "Tăng cường dữ liệu human-in-the-loop từ data/feedback/",
        "Fine-tune chuyên biệt cho giày dép",
        "Mở rộng sang nhiều loại sản phẩm thương mại điện tử",
    ],
    "Threats": [
        "Thay đổi API/weights của TripoSR hoặc Depth-Anything",
        "Tài nguyên GPU hạn chế khi scale lên nhiều ảnh",
        "Chất lượng ảnh người dùng khác biệt gây sai lệch đầu ra",
    ],
}
ROADMAP_ITEMS = [
    "Thu thập feedback, xây dựng dataset lỗi trong data/feedback/",
    "Fine-tune TripoSR/Depth-Anything trên domain giày dép",
    "Nâng độ phân giải Marching Cubes cao hơn (384³+)",
    "Thêm multi-view backup path nếu single-image chưa đủ",
    "Tối ưu giao diện Streamlit và cơ chế cache toàn hệ thống",
]


def save_fig(fig, name):
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def create_traditional_vs_ai_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(PROCESSING_TIMES.keys())
    times = list(PROCESSING_TIMES.values())
    bars = ax.bar(labels, times, color=["#ff6666", "#66b3ff", "#99ff99"])

    ax.set_yscale("log")
    ax.set_ylabel("Thời gian (giây)")
    ax.set_title("So sánh tốc độ Truyền thống ↔ AI (dữ liệu dự án)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, time_val in zip(bars, times):
        text = f"{time_val:.5f}s" if time_val < 0.1 else f"{time_val:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2, text,
                ha="center", va="bottom", fontweight="bold")

    ax.annotate(
        "Truyền thống: hàng giờ -> nhiều nghìn giây",
        xy=(0.5, 0.15), xycoords="axes fraction",
        xytext=(0.1, 0.02), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0", ec="#666666", alpha=0.8),
    )

    return save_fig(fig, "slide2_traditional_vs_ai")


def create_input_output_chart(image_path):
    img = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Ảnh đầu vào mẫu")

    ax_diag = fig.add_subplot(gs[1])
    ax_diag.axis("off")
    ax_diag.set_title("Input → Output / Dự án")

    boxes = [
        (0.10, 0.72, "Ảnh 2D"),
        (0.10, 0.50, "Tiền xử lý CVIP"),
        (0.10, 0.28, "Depth-Anything-V2"),
        (0.10, 0.06, "TripoSR + UV Texture"),
    ]
    for x, y, text in boxes:
        rect = patches.FancyBboxPatch((x, y), 0.8, 0.16, boxstyle="round,pad=0.15",
                                      edgecolor="#333333", facecolor="#f7f7f7", lw=1.8)
        ax_diag.add_patch(rect)
        ax_diag.text(x + 0.4, y + 0.08, text, ha="center", va="center", fontsize=11, weight="bold")

    for i in range(len(boxes) - 1):
        x = 0.5
        y0 = boxes[i][1]
        y1 = boxes[i + 1][1] + 0.16
        ax_diag.annotate("",
                        xy=(x, y1), xytext=(x, y0),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#444444"))

    return save_fig(fig, "slide3_input_output")


def create_system_architecture_chart():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title("Kiến trúc tổng thể hệ thống", fontsize=16, pad=20)

    regions = [
        (0.05, 0.58, 0.28, 0.25, "Giao diện / Tải ảnh"),
        (0.37, 0.58, 0.28, 0.25, "Cache + SHA-256"),
        (0.69, 0.58, 0.28, 0.25, "Tiền xử lý ảnh CVIP"),
        (0.05, 0.18, 0.28, 0.25, "Ước lượng chiều sâu"),
        (0.37, 0.18, 0.28, 0.25, "Dựng Mesh 3D TripoSR"),
        (0.69, 0.18, 0.28, 0.25, "Xuất GLB / Feedback"),
    ]

    for x, y, w, h, text in regions:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      edgecolor="#222222", facecolor="#e7f1fd", lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, weight="bold")

    arrow_specs = [
        ((0.2, 0.58), (0.2, 0.43)),
        ((0.58, 0.58), (0.58, 0.43)),
        ((0.8, 0.58), (0.8, 0.43)),
        ((0.2, 0.43), (0.2, 0.33)),
        ((0.58, 0.43), (0.58, 0.33)),
        ((0.8, 0.43), (0.8, 0.33)),
    ]
    for start, end in arrow_specs:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    ax.text(0.5, 0.95, "Flow từ ảnh đầu vào tới mô hình 3D", ha="center", va="center", fontsize=13)
    return save_fig(fig, "slide5_system_architecture")


def create_preprocessing_workflow_chart(image_path):
    img = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.12)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Ảnh gốc mẫu")

    ax_flow = fig.add_subplot(gs[1])
    ax_flow.axis("off")
    ax_flow.set_title("Workflow tiền xử lý ảnh thực tế", pad=20)

    steps = [
        "1. Tách nền (rembg)",
        "2. Crop + Resize foreground",
        "3. Lọc Fourier + Wavelet",
        "4. Mask + Edge Detection",
        "5. Chuẩn hoá nền + xuất ảnh sạch",
    ]
    for i, step in enumerate(steps):
        y = 0.8 - i * 0.15
        rect = patches.FancyBboxPatch((0.08, y), 0.84, 0.13,
                                      boxstyle="round,pad=0.2", facecolor="#fff5cc",
                                      edgecolor="#d28c18", lw=1.8)
        ax_flow.add_patch(rect)
        ax_flow.text(0.5, y + 0.065, step, ha="center", va="center", fontsize=11)
        if i < len(steps) - 1:
            ax_flow.annotate("", xy=(0.5, y - 0.02), xytext=(0.5, y - 0.10),
                             arrowprops=dict(arrowstyle="->", lw=2, color="#444444"))

    return save_fig(fig, "slide6_preprocessing_workflow")


def create_cvip_timeline_chart():
    steps = [
        "Color Spaces",
        "Fourier Filter",
        "Wavelet Denoise",
        "Geometric Resize",
        "Morphology",
        "Edge Detection",
        "Segmentation",
        "Feature Detection",
        "Bounding Box Crop",
        "Depth Estimation",
        "Point Cloud",
        "TripoSR Mesh",
        "UV Mapping & Texture",
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(range(len(steps)), [1] * len(steps), color="#7ab8ff")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([f"{i+1}. {step}" for i, step in enumerate(steps)], fontsize=10)
    ax.set_xticks([])
    ax.set_title("Timeline 13 bước CVIP trong Pipeline", fontsize=16)
    ax.invert_yaxis()
    for i in range(len(steps)):
        ax.text(0.5, i, "", va="center")
    return save_fig(fig, "slide7_cvip_timeline")


def create_depth_anything_architecture_chart():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title("Kiến trúc Depth-Anything-V2 (Depth Estimation)", fontsize=16, pad=20)

    boxes = [
        (0.05, 0.60, 0.25, 0.30, "RGB 2D (518×518)"),
        (0.38, 0.60, 0.25, 0.30, "ViT-S Backbone"),
        (0.71, 0.60, 0.25, 0.30, "DPT Decoder"),
        (0.38, 0.16, 0.25, 0.25, "Depth Map [0,1]") ,
    ]
    for x, y, w, h, text in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      facecolor="#d9edf7", edgecolor="#31708f", lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, weight="bold")

    arrows = [((0.30, 0.75), (0.38, 0.75)), ((0.63, 0.75), (0.71, 0.75)), ((0.50, 0.60), (0.50, 0.41))]
    for src, dst in arrows:
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="->", lw=2, color="#444444"))
    return save_fig(fig, "slide8_depth_anything_architecture")


def create_tripossr_architecture_chart():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title("Kiến trúc TripoSR (LRM) trong dự án", fontsize=16, pad=20)

    steps = [
        (0.10, 0.60, "Ảnh tách nền RGBA"),
        (0.40, 0.60, "Image-to-Triplane Transformer"),
        (0.70, 0.60, "Implicit Field + MLP"),
        (0.40, 0.30, "Marching Cubes \n(resolution=384)"),
        (0.70, 0.10, "Mesh trắng + UV bake"),
    ]
    for x, y, text in steps:
        rect = patches.FancyBboxPatch((x, y), 0.23, 0.20, boxstyle="round,pad=0.2",
                                      facecolor="#fde9d9", edgecolor="#a65d12", lw=1.8)
        ax.add_patch(rect)
        ax.text(x + 0.115, y + 0.10, text, ha="center", va="center", fontsize=11, weight="bold")

    arrows = [((0.33, 0.70), (0.40, 0.70)), ((0.63, 0.70), (0.70, 0.70)), ((0.52, 0.50), (0.52, 0.40)), ((0.63, 0.30), (0.70, 0.30))]
    for src, dst in arrows:
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    return save_fig(fig, "slide9_tripossr_architecture")


def create_real_workflow_chart():
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = list(MODULE_TIMES.keys())
    times = list(MODULE_TIMES.values())
    bars = ax.bar(steps, times, color=["#f5c09a", "#f5de64", "#85bced"], edgecolor="#444444")
    ax.set_ylabel("Thời gian (giây)")
    ax.set_title("Workflow xử lý ảnh thực tế của module AI", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{t:.2f}s", ha="center", va="bottom", fontweight="bold")
    return save_fig(fig, "slide10_real_workflow")


def create_confusion_matrix_chart():
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(CONFUSION_MATRIX, cmap="Blues")

    ax.set_xticks(np.arange(len(CONFUSION_LABELS)))
    ax.set_yticks(np.arange(len(CONFUSION_LABELS)))
    ax.set_xticklabels(CONFUSION_LABELS, color="black")
    ax.set_yticklabels(CONFUSION_LABELS, color="black")
    ax.set_xlabel("Predict", color="black")
    ax.set_ylabel("Thực tế", color="black")
    ax.set_title("Confusion Matrix (trên tập Test)", pad=20, color="black")

    for i in range(CONFUSION_MATRIX.shape[0]):
        for j in range(CONFUSION_MATRIX.shape[1]):
            ax.text(j, i, int(CONFUSION_MATRIX[i, j]), ha="center", va="center", color="black", fontsize=18, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Số lượng mẫu", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    return save_fig(fig, "confusion_matrix")


def create_evaluation_radar_chart():
    categories = list(EVALUATION_SCORES.keys())
    values = list(EVALUATION_SCORES.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#1f77b4", linewidth=2)
    ax.fill(angles, values, color="#1f77b4", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Đánh giá dự án qua các tiêu chí chính", pad=20, fontsize=14)
    return save_fig(fig, "slide11_evaluation_radar")


def create_processing_time_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(PROCESSING_TIMES.keys())
    times = list(PROCESSING_TIMES.values())
    bars = ax.bar(labels, times, color=["#ff6666", "#66b3ff", "#99ff99"])
    ax.set_ylabel("Thời gian xử lý (giây)")
    ax.set_title("Biểu đồ thời gian xử lý từng giai đoạn", fontsize=16)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for bar, time_val in zip(bars, times):
        text = f"{time_val:.5f}s" if time_val < 0.1 else f"{time_val:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1, text,
                ha="center", va="bottom", fontweight="bold")
    return save_fig(fig, "slide12_processing_time")


def create_dataset_chart():
    labels = list(DATASET_STATS.keys())
    values = list(DATASET_STATS.values())
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct=lambda pct: f"{int(round(pct))}%" if pct else "0%",
                                      startangle=90, colors=["#66c2a5", "#fc8d62", "#8da0cb"], textprops=dict(color="black"))
    ax.set_title("Dataset của dự án: 100% Test – Zero-shot", fontsize=14)
    ax.axis("equal")
    ax.legend(wedges, [f"{label} ({value})" for label, value in zip(labels, values)], loc="lower left")
    return save_fig(fig, "slide13_dataset")


def create_swot_chart():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    ax.set_title("Bảng SWOT của dự án", fontsize=18, pad=25)

    categories = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
    x_positions = [0.05, 0.525]
    y_positions = [0.55, 0.05]
    width = 0.43
    height = 0.38
    colors = ["#d9f0d3", "#fde0dd", "#d0e0f0", "#f1d9d9"]
    for (i, cat), color in zip(enumerate(categories), colors):
        x = x_positions[i % 2]
        y = y_positions[i // 2]
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3",
                                      facecolor=color, edgecolor="#333333", lw=1.8)
        ax.add_patch(rect)
        ax.text(x + 0.02, y + height - 0.05, cat, fontsize=14, weight="bold", va="top")
        for j, item in enumerate(SWOT_CONTENT[cat]):
            ax.text(x + 0.03, y + height - 0.13 - j * 0.10, f"• {item}", fontsize=11, va="top")
    return save_fig(fig, "slide14_swot")


def create_roadmap_chart():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title("Roadmap phát triển dự án", fontsize=18, pad=20)

    for idx, item in enumerate(ROADMAP_ITEMS, start=1):
        y = 0.85 - (idx - 1) * 0.16
        circle = patches.Circle((0.08, y), 0.03, color="#4a90e2")
        ax.add_patch(circle)
        ax.text(0.08, y, str(idx), color="white", ha="center", va="center", fontsize=12, weight="bold")
        ax.text(0.14, y, item, ha="left", va="center", fontsize=12)
        if idx < len(ROADMAP_ITEMS):
            ax.annotate("", xy=(0.08, y - 0.08), xytext=(0.08, y - 0.13),
                        arrowprops=dict(arrowstyle="-|>", color="#4a90e2", lw=1.8))

    return save_fig(fig, "slide15_roadmap")


def main():
    image_path = ROOT / "data" / "uploads" / "anhdemo.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh mẫu tại {image_path}")

    chart_paths = []
    chart_paths.append(create_traditional_vs_ai_chart())
    chart_paths.append(create_input_output_chart(image_path))
    chart_paths.append(create_system_architecture_chart())
    chart_paths.append(create_preprocessing_workflow_chart(image_path))
    chart_paths.append(create_cvip_timeline_chart())
    chart_paths.append(create_depth_anything_architecture_chart())
    chart_paths.append(create_tripossr_architecture_chart())
    chart_paths.append(create_real_workflow_chart())
    chart_paths.append(create_confusion_matrix_chart())
    chart_paths.append(create_evaluation_radar_chart())
    chart_paths.append(create_processing_time_chart())
    chart_paths.append(create_dataset_chart())
    chart_paths.append(create_swot_chart())
    chart_paths.append(create_roadmap_chart())

    pdf_path = OUT_DIR / "presentation_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for png_path in chart_paths:
            img = Image.open(png_path)
            fig = plt.figure(figsize=(img.width / 100, img.height / 100))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved presentation charts:")
    for path in chart_paths:
        print(f" - {path}")
    print(f"Saved combined PDF: {pdf_path}")


if __name__ == "__main__":
    main()
