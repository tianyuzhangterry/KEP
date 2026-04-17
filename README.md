# 🧠 KnowledgeEditPlatform

**KnowledgeEditPlatform**  
一个面向大语言模型（LLM）与多模态大模型（MLLM）的一体化知识编辑验证平台，  
旨在统一实现、复现、对比主流知识编辑方法，并在统一评测协议下生成可比实验结果。

本项目结构将知识编辑问题抽象为 **Editor、Method、Evaluate** 三层架构，并在此基础上扩展支持多模态场景与连续编辑协议，从而实现方法插件化、评测标准化与实验可复现性。

平台同时支持：

- 📌 事实性知识编辑（Factual Knowledge Editing）
- 📌 多模态模型知识编辑（MultiModal Model Editing）
- 📌 参数修改类与参数保持类方法
- 📌 单次 / 批量 / 连续 编辑协议

---

## 🚀 主要特性

- 🧩 **统一架构设计**  
  将模型适配、方法注册与评测分离，Editor、Method 与 Evaluate 层模块化设计，易于扩展。

- 🔍 **多方法集成**  
  内置并复现多种经典编辑方法（如 ROME、MEMIT、MEND、IKE、SERAC）以及支持自定义方法接入。

- 📊 **统一评测协议**  
  支持 CounterFact、ZsRE 等标准基准，与自定义多模态任务评测协议，一致性对比。

- 🔁 **连续编辑支持**  
  支持多轮编辑执行，自动记录各轮指标趋势，便于分析模型稳定性。

- 🧪 **Auto Reporting**  
  自动生成主结果表、评测指标图表、趋势曲线等对比结果，方便论文与报告输出。

---

## 🧠 核心设计思想

① **Editor 层**：  
代表某类知识编辑场景，比如文本事实知识、视觉语言知识等。  
负责接收目标模型、目标知识及其他超参，调用方法层实现。

② **Method 层**：  
具体的知识编辑算法实现，例如参数定位修改（ROME、MEMIT）、参数保持（IKE、SERAC、MEND）。  
每个方法实现都需继承统一接口，便于插件化接入。

③ **Evaluate 层**：  
实现统一评测指标，在不同任务与数据协议下进行可靠性（Reliability）、泛化（Generalization）、局部性（Locality）、效率（Efficiency）等指标计算。

---


## ⚙️ Environment Setup

We recommend using `conda` for environment management.

### 1. Create Environment

```bash
conda create -n keplatform python=3.10
conda activate keplatform

### 2. Install Dependencies
```bash
pip install -r requirements.txt

### 3. Install Core Libraries
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate

## 📦 Installation

```bash
git clone https://github.com/yourname/KnowledgeEditPlatform.git
cd KnowledgeEditPlatform
pip install -e .





# 🧠 Task Definition

```markdown
## 🧠 Task Definition

Given a pretrained model $f_\theta$, knowledge editing aims to modify its behavior on specific edit pairs $(x_e, y_e)$:

- $x_e$: input prompt  
- $y_e$: target output  

while preserving performance on unrelated inputs.

### Editing Settings

- **Single Editing**: apply one edit independently  
- **Batch Editing**: apply multiple edits simultaneously  
- **Sequential Editing**: continuous updates on the same model 

## 📊 Evaluation

We adopt a unified evaluation protocol:

- **Reliability**: success rate on edited knowledge  
- **Generalization**: performance on paraphrased inputs  
- **Locality**: impact on unrelated knowledge  
- **Efficiency**: time and memory cost  

### Output Format

```json
{
  "post": {
    "reliability": 0.92,
    "generalization": 0.88,
    "locality": 0.97
  }
}

# 📊 Visualization

Example：
<img width="1034" height="570" alt="image" src="https://github.com/user-attachments/assets/eef4dff4-5184-4af9-be31-385d3dea1235" />
