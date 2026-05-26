# 材料孔洞面积分析系统

这是一个基于 Streamlit 的材料图像孔洞面积分析 Web 应用，支持批量上传 SEM/TEM/金相显微镜图像，自动识别孔洞并计算孔洞面积占比。

## 主要功能

- 批量上传多张图像
- 自动识别并标注孔洞边缘
- 计算每张图像的孔洞面积占比、孔洞数量、平均孔洞面积
- 展示分析结果概览、详细结果和汇总表格
- 可选生成交互式孔洞占比分布散点图

## 运行步骤

1. 创建并激活虚拟环境
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
3. 启动 Streamlit 应用
   ```bash
   streamlit run app.py
   ```
4. 在浏览器中访问
   ```text
   http://localhost:8501
   ```

## 使用说明

1. 上传图像文件（支持 JPG/JPEG/PNG/BMP/TIFF/TIF）
2. 设置二值化阈值、最小孔洞面积和形态学核大小
3. 点击「开始分析」执行批量图像处理
4. 查看结果概览、详细结果和汇总表格
5. 可选勾选「生成孔洞占比散点图」并输入图像代称，生成交互式图表

## 注意

- 如果图表 PNG 下载无法生成，请确保 `matplotlib` 已正确安装。
- 应用使用 matplotlib 生成静态 PNG 导出，不再依赖 kaleido。
- Streamlit 部署环境（如 Streamlit Cloud）建议使用 headless OpenCV：
  ```bash
  pip install opencv-python-headless
  ```
- 形态学核大小应为奇数，否则分析前会提示调整。
