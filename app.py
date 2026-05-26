import io
import os
import shutil
import tempfile
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV 未安装。请确保 requirements.txt 中包含 `opencv-python-headless`，并重新安装依赖。"
    ) from exc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


def generate_plot_png(labels, ratios, counts):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    x_positions = list(range(len(labels)))
    sizes = [max(60, c * 10) for c in counts]

    ax.scatter(x_positions, ratios, s=sizes, alpha=0.7, edgecolors="w", linewidth=0.5)
    ax.set_title("孔洞面积占比分布图", fontsize=16)
    ax.set_xlabel("图像代称", fontsize=12)
    ax.set_ylabel("孔洞面积占比 (%)", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.3)

    for x, y in zip(x_positions, ratios):
        ax.text(x, y + 0.5, f"{y:.2f}%", ha="center", va="bottom", fontsize=9)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


# 1. 分析函数定义

def analyze_holes(image_path, threshold, min_area, kernel_size):
    """分析单张图像中的孔洞，并返回标注图像、孔洞占比、孔洞数量和每个孔洞面积列表。"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    hole_areas = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append(contour)
            hole_areas.append(float(area))

    total_area = float(img.shape[0] * img.shape[1])
    hole_area_sum = float(sum(hole_areas))
    hole_ratio = float((hole_area_sum / total_area) * 100.0) if total_area > 0 else 0.0

    annotated = img.copy()
    cv2.drawContours(annotated, valid_contours, -1, (0, 255, 0), 2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, hole_ratio, len(hole_areas), hole_areas


def image_to_bytes(image_array):
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def init_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = ""


def clear_analysis_state():
    if st.session_state.get("temp_dir") and os.path.isdir(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    st.session_state.analysis_results = []
    st.session_state.analysis_done = False
    st.session_state.temp_dir = ""


def save_uploaded_file(uploaded_file, save_dir):
    save_path = os.path.join(save_dir, uploaded_file.name)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    uploaded_file.seek(0)
    return save_path


# 3. Streamlit 页面配置
st.set_page_config(
    page_title="材料孔洞面积分析系统",
    layout="wide",
    initial_sidebar_state="auto",
)

# 4. 初始化 session_state
init_session_state()

# 5. UI 构建（按区域1-6）
st.title("材料孔洞面积分析系统")
st.markdown(
    """
    📋 **使用步骤**：① 上传图像 → ② 设置分析参数 → ③ 点击开始分析 → ④ 查看结果 → ⑤ 可选：生成孔洞占比散点图
    """
)

st.write("---")

# 区域2：图像上传
uploaded_files = st.file_uploader(
    "📁 拖拽或点击上传图像文件",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if uploaded_files is not None:
    current_file_names = [f.name for f in uploaded_files]
    if current_file_names != st.session_state.uploaded_files:
        clear_analysis_state()
        st.session_state.uploaded_files = current_file_names

    if len(uploaded_files) > 0:
        st.write(f"已上传图像总数：**{len(uploaded_files)}**")

        preview_cols = st.columns(4)
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                with preview_cols[idx % 4]:
                    display_name = (
                        uploaded_file.name
                        if len(uploaded_file.name) <= 30
                        else f"{uploaded_file.name[:27]}..."
                    )
                    st.image(img, width=200)
                    st.caption(display_name)
            except Exception:
                with preview_cols[idx % 4]:
                    st.warning(f"无法预览：{uploaded_file.name}")

# 区域3：分析参数设置
with st.expander("⚙️ 高级分析参数设置", expanded=False):
    threshold = st.slider(
        "二值化阈值",
        min_value=0,
        max_value=255,
        value=127,
        step=1,
        help="像素灰度值低于此阈值的区域将被视为孔洞",
    )
    min_area = st.number_input(
        "最小孔洞面积过滤 (像素)",
        min_value=0,
        value=50,
        step=10,
        help="面积小于此值的区域将被视为噪声并过滤",
    )
    kernel_size = st.number_input(
        "形态学闭运算核大小",
        min_value=1,
        max_value=21,
        value=5,
        step=2,
        help="用于填充孔洞边缘小间隙，必须为奇数",
    )
    if kernel_size % 2 == 0:
        st.warning("形态学核大小必须为奇数，请调整后重新分析。")

st.write("---")

# 区域4：分析触发与进度
analysis_placeholder = st.empty()
progress_placeholder = st.empty()

start_analysis = st.button("🔍 开始分析", type="primary", use_container_width=True)

if start_analysis:
    if not uploaded_files:
        st.warning("请先上传至少一张图像，然后再开始分析。")
    elif kernel_size % 2 == 0:
        st.warning("请将形态学核大小设置为奇数后再次开始分析。")
    else:
        if not st.session_state.temp_dir:
            st.session_state.temp_dir = tempfile.mkdtemp(prefix="hole_area_analysis_")

        analysis_results = []
        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            status = f"正在处理：{uploaded_file.name} ({idx + 1}/{total_files})"
            analysis_placeholder.info(status)
            progress_placeholder.progress(int((idx / total_files) * 100))

            try:
                save_path = save_uploaded_file(uploaded_file, st.session_state.temp_dir)
                annotated_rgb, hole_ratio, hole_count, hole_areas = analyze_holes(
                    save_path, threshold, min_area, kernel_size
                )
                uploaded_file.seek(0)
                original_bytes = uploaded_file.read()

                analysis_results.append(
                    {
                        "file_name": uploaded_file.name,
                        "tmp_path": save_path,
                        "original_bytes": original_bytes,
                        "annotated_image": annotated_rgb,
                        "hole_ratio": round(hole_ratio, 4),
                        "hole_count": hole_count,
                        "hole_areas": hole_areas,
                        "avg_hole_area": round(float(np.mean(hole_areas)) if hole_areas else 0.0, 2),
                    }
                )
            except Exception as exc:
                st.warning(f"文件 {uploaded_file.name} 处理失败：{exc}")
            finally:
                progress_placeholder.progress(int(((idx + 1) / total_files) * 100))

        st.session_state.analysis_results = analysis_results
        st.session_state.analysis_done = len(analysis_results) > 0
        if st.session_state.analysis_done:
            st.success("分析完成！请查看下面的结果概览和详细展示。")
        else:
            st.error("分析未能生成有效结果，请检查输入图像或参数设置。")

if st.session_state.analysis_done and st.session_state.analysis_results:
    results = st.session_state.analysis_results

    # 区域5a：结果概览
    st.write("---")
    st.subheader("结果概览")

    total_images = len(results)
    average_ratio = round(float(np.mean([res["hole_ratio"] for res in results])), 4)
    max_item = max(results, key=lambda item: item["hole_ratio"])
    total_holes = sum(res["hole_count"] for res in results)

    metric_cols = st.columns(4)
    metric_cols[0].metric("分析图像总数", total_images)
    metric_cols[1].metric("平均孔洞占比", f"{average_ratio:.2f}%")
    metric_cols[2].metric("最大孔洞占比", f"{max_item['hole_ratio']:.2f}%")
    metric_cols[2].caption(f"图像：{max_item['file_name']}")
    metric_cols[3].metric("总识别孔洞数", total_holes)

    # 区域5b：详细结果展示
    st.write("---")
    st.subheader("详细结果展示")

    for idx, res in enumerate(results, start=1):
        with st.container():
            st.markdown(f"#### {idx}. {res['file_name']}")
            left_col, right_col = st.columns([1, 2])

            with left_col:
                try:
                    original_image = Image.open(io.BytesIO(res["original_bytes"]))
                    st.image(original_image, caption="原始图像", width=300)
                except Exception:
                    st.warning("无法显示原始图像预览。")

            with right_col:
                st.image(res["annotated_image"], caption="标注后孔洞图像", width=600)
                st.markdown(f"**孔洞占比：** {res['hole_ratio']:.2f}%")
                st.markdown(f"**孔洞数量：** {res['hole_count']}")
                st.markdown(f"**平均孔洞面积：** {res['avg_hole_area']:.2f} 像素")
                download_bytes = image_to_bytes(res["annotated_image"])
                st.download_button(
                    "下载标注后的图像",
                    data=download_bytes,
                    file_name=f"{Path(res['file_name']).stem}_annotated.png",
                    mime="image/png",
                )

    # 区域5c：汇总表格
    st.write("---")
    st.subheader("汇总表格")

    summary_data = []
    for index, res in enumerate(results, start=1):
        summary_data.append(
            {
                "序号": index,
                "文件名": res["file_name"],
                "孔洞数量": res["hole_count"],
                "孔洞面积占比(%)": round(res["hole_ratio"], 4),
                "平均孔洞面积(像素)": res["avg_hole_area"],
            }
        )

    df_summary = pd.DataFrame(summary_data).sort_values("孔洞面积占比(%)", ascending=False)
    st.dataframe(
        df_summary,
        column_config={
            "序号": st.column_config.NumberColumn("序号"),
            "文件名": st.column_config.TextColumn("文件名"),
            "孔洞数量": st.column_config.NumberColumn("孔洞数量"),
            "孔洞面积占比(%)": st.column_config.NumberColumn("孔洞面积占比(%)", format="%.4f"),
            "平均孔洞面积(像素)": st.column_config.NumberColumn("平均孔洞面积(像素)", format="%.2f"),
        },
        hide_index=True,
    )

    # 区域6：散点图生成（可选）
    st.write("---")
    st.subheader("散点图生成（可选）")

    show_scatter = st.checkbox("📊 生成孔洞占比散点图", value=False)
    scatter_labels = []
    if show_scatter:
        # 在表单中仅收集用户输入（文本框），不要在表单内生成图表或下载按钮
        with st.form("scatter_form"):
            st.write("为每张图像输入横轴标签：")
            input_keys = []
            for idx, res in enumerate(results):
                default_label = Path(res["file_name"]).stem
                key = f"scatter_label_{idx}"
                _ = st.text_input(
                    f"请输入【{res['file_name']}】的代称",
                    value=default_label,
                    key=key,
                )
                input_keys.append(key)

            submitted = st.form_submit_button("生成散点图")

            if submitted:
                # 从 session_state 中读取刚填写的标签并存储，用于表单外绘图与下载
                labels = [st.session_state.get(k, Path(results[i]["file_name"]).stem) for i, k in enumerate(input_keys)]
                st.session_state["scatter_labels"] = labels
                st.session_state["scatter_requested"] = True

        # 表单外渲染图表与下载按钮，避免在表单内部使用下载控件
        if st.session_state.get("scatter_requested"):
            labels = st.session_state.get("scatter_labels", [Path(r["file_name"]).stem for r in results])
            scatter_df = pd.DataFrame(
                {
                    "代称": labels,
                    "孔洞占比(%)": [res["hole_ratio"] for res in results],
                    "孔洞数量": [res["hole_count"] for res in results],
                    "原文件名": [res["file_name"] for res in results],
                }
            )
            fig = px.scatter(
                scatter_df,
                x="代称",
                y="孔洞占比(%)",
                size="孔洞数量",
                hover_data=["原文件名", "孔洞占比(%)", "孔洞数量"],
                title="孔洞面积占比分布图",
            )
            fig.update_layout(
                xaxis_title="图像代称",
                yaxis_title="孔洞面积占比 (%)",
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            try:
                png_bytes = generate_plot_png(
                    labels,
                    [res["hole_ratio"] for res in results],
                    [res["hole_count"] for res in results],
                )
                st.download_button(
                    "下载图表为 PNG",
                    data=png_bytes,
                    file_name="hole_ratio_scatter.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(
                    f"PNG 导出失败：{e}。请确认 matplotlib 已安装，并在部署环境中包含 matplotlib。"
                )
                html = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    "下载图表为 HTML (回退)",
                    data=html,
                    file_name="hole_ratio_scatter.html",
                    mime="text/html",
                )
