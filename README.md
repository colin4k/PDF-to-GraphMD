# PDF-to-GraphMD

**自动化知识图谱构建系统** - 将PDF电子书转换为Obsidian兼容的知识网络

## 项目概述

PDF-to-GraphMD是一个端到端的自动化系统，能够将非结构化的PDF文档转换为结构化的、相互链接的知识图谱。系统生成与Obsidian笔记软件完全兼容的Markdown文件集合，每个知识点为独立文件，知识点间通过`[[维基链接]]`语法建立关系。

## 核心特性

### 🔄 四阶段处理流程
1. **摄取与解析** - 使用MinerU进行高保真PDF内容提取
2. **知识提取** - 支持LLM和NLP两种可配置的提取方法
3. **图谱构建** - 智能实体规范化和关系验证
4. **输出生成** - 生成Obsidian兼容的Markdown文件

### 📄 强大的PDF解析能力
- **完整文档结构** - 保留标题、段落、列表、多栏布局
- **表格提取** - 自动转换为Markdown/HTML格式
- **公式识别** - 数学公式转换为LaTeX格式
- **图像处理** - 提取图像及图注文字
- **OCR支持** - 自动处理扫描版PDF和文本识别

### 🧠 灵活的知识提取
- **LLM路径** - 基于大语言模型的智能提取
  - 结构化提示工程
  - 强制JSON输出
  - 可配置本体定义
- **NLP路径** - 基于spaCy的专业化处理
  - 命名实体识别(NER)
  - 依存句法分析
  - 基于规则的关系提取

### 🎯 智能图谱构建
- **实体规范化** - 自动消除歧义和重复
- **关系验证** - 确保图谱完整性和一致性
- **增量处理** - 支持新文档融入现有知识库

### 📝 Obsidian优化输出
- **YAML Frontmatter** - 结构化元数据
- **维基链接** - `[[实体名称]]`语法
- **关系分类** - 按类型组织的相关链接
- **嵌入内容** - 表格、公式、图像完整保留

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/pdf-to-graphmd.git
cd pdf-to-graphmd

# 安装依赖
pip install -r requirements.txt

# 或使用pip安装
pip install pdf-to-graphmd
```

### 基本使用

```bash
# 处理单个PDF文件
python -m pdf_to_graphmd --input document.pdf

# 处理目录中所有PDF
python -m pdf_to_graphmd --input-dir ./documents/

# 使用自定义配置
python -m pdf_to_graphmd --input document.pdf --config config.yaml

# 指定提取方法
python -m pdf_to_graphmd --input document.pdf --method nlp

# 自定义输出目录
python -m pdf_to_graphmd --input document.pdf --output ./my_vault/
```

### Python API使用

```python
from pdf_to_graphmd import process_pdf_file, create_processor

# 简单处理
result = process_pdf_file("document.pdf")

# 高级使用
processor = create_processor("config.yaml")
result = processor.process_single_pdf("document.pdf")

print(f"生成了 {len(result.obsidian_notes)} 个笔记")
print(f"提取了 {len(result.knowledge_graph.entities)} 个实体")
```

## 配置

### 生成配置文件

```bash
python -m pdf_to_graphmd --generate-config config.yaml
```

### 配置示例

```yaml
extraction_method: "llm"  # 或 "nlp"

# MinerU设置
mineru:
  language: "ch"
  use_gpu: true
  output_formats: ["markdown", "json"]

# LLM设置
llm:
  model_name: "gpt-3.5-turbo"
  api_key: "your-api-key"
  temperature: 0.1

# 本体定义
ontology:
  entity_types:
    - "Person"
    - "Organization" 
    - "Concept"
    - "Theory"
    - "Method"
  relation_types:
    - "defined_as"
    - "part_of"
    - "related_to"
    - "proposed_by"

# 输出设置
output:
  output_dir: "./obsidian_vault"
  include_images: true
  include_tables: true
  include_formulas: true
```

## 系统架构

```
PDF文档 → MinerU解析 → 知识提取 → 图谱构建 → Obsidian输出
   ↓         ↓           ↓          ↓         ↓
原始PDF → 结构化内容 → 实体关系 → 规范图谱 → Markdown文件
```

### 核心模块

- **`parsers/`** - PDF解析模块（MinerU集成）
- **`extractors/`** - 知识提取模块（LLM/NLP）
- **`graph/`** - 图谱构建与规范化
- **`output/`** - Obsidian文件生成
- **`config.py`** - 配置管理
- **`models.py`** - 数据模型定义

## 依赖要求

### 核心依赖
- **MinerU** (magic-pdf) - PDF解析引擎
- **OpenAI** - LLM API支持
- **spaCy** - NLP处理
- **PyTorch** - 深度学习支持

### 可选依赖
- **CUDA** - GPU加速（推荐）
- **自定义NER模型** - 领域特定实体识别

## 许可证

MIT License
