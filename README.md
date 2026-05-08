🎨 Manga Localizer Pro: 工业级漫画自动化翻译流水线

本项目是一套基于 Manga-Translator-UI (MTU) 深度定制的工业级漫画自动化翻译工作流。通过集成 XGBoost 物理审计哨兵与 动态 GBNF 语法约束，彻底解决了本地大模型在漫画翻译中的幻觉、扩写及噪点误读等痛点。
🔥 核心技术亮点
🛡️ 1. XGBoost 审核 (MangaSentinelJP)

不再依赖单一的 OCR 置信度，我们构建了一个基于 22 维物理指纹的分类模型：

    特征维度：涵盖空洞率 (Ratio)、语言熵 (平/片假名占比)、相对页面中值偏移 (P_Dev)、物理方向、几何位置等。

    性能：经过 Optuna 调参，在 30,000+ 条真实漫画数据上实现了 96%+ 的噪声拦截准确率。

    物理旁路：自动识别拟声词 (SFX) 与 OCR 噪点 (如 コ君, j, E9)，执行物理隔离，确保“屎”不入 LLM，从根源杜绝幻觉。

🧩 2. 动态 GBNF 语法锁死 (Strict Formatting)

通过物理层面的 GBNF (Guided Batch Normalization Form) 约束 LLM 输出：

    行数对齐：根据输入行数动态生成 line{N} 语法，强制模型在完成翻译后立即停止。

    结构保护：物理锁死 <textarea> 标签，彻底根治了本地模型常见的“自言自语”和“剧情续写”问题。

📐 3. 物理层渲染修复 (Coordinate Self-Healing)

    尺度不变性：支持 2x 超分（MangaJaNai）环境下自动同步 1x 渲染坐标。

    几何去畸变：自动检测并删除退化多边形（点数 < 3），防止 OpenCV fillPoly 断言崩溃。

    类型锁死：强制执行 int(round()) 转换，确保 100% 渲染成功率。

🏗️ 自动化流水线架构

项目采用 四阶段物理隔离 设计，确保流程稳健：

    Stage 1: Export (提取)
    调用 MTU 执行 MangaJaNai 2x 超分，导出原始 JSON 元数据与 TXT 语料。

    Stage 2: Audit & Translate (审计与翻译)
    XGBoost 对 TXT 每一行进行物理审计。

        Noise

                
        →→
              

        物理旁路，回填原文。

        Valid

                
        →→


        送入 LLM，基于 V6.5 纯净版提示词翻译。

    Stage 3: Patching (补丁)
    执行物理降维补丁，同步 2x 坐标到 1x 空间，保持 Base64 蒙版完整。

    Stage 4: Render (渲染)
    调用 MTU load_text 模式，实现 1:1 精准嵌字。

🚀 性能优化

    并行加速：支持 --batch_concurrent 异步 OCR 处理，配合单线程稳健翻译，兼顾速度与质量。

🛠️ 快速开始
环境依赖

    Python 3.10+

    XGBoost, Scipy, Pandas

    Manga-Translator-UI (已安装并配置好 local 环境)

    OpenAI 兼容接口的本地模型后端 (如 llama.cpp)

运行

    执行主程序：

code Bash

python auto_tran.py



📜 许可证

本项目仅供学习与研究使用，请遵守当地法律法规。

Developed with ❤️ for the Manga Translation Community.
