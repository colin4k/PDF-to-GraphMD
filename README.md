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
- **LLM路径** - 支持多种大语言模型API
  - **Google原生API** - 直接调用Google AI Studio/Gemini API
  - **OpenAI兼容API** - 支持OpenRouter等代理服务
  - 结构化提示工程和强制JSON输出
  - 可配置本体定义
- **NLP路径** - 基于spaCy的专业化处理
  - 命名实体识别(NER)
  - 依存句法分析
  - 基于规则的关系提取

### 🤖 多模型API支持
- **Google AI Studio** - 原生Gemini 2.5 Pro支持
  - 更低延迟和更高稳定性
  - 完整功能支持
  - 原生JSON模式
- **OpenAI兼容** - 支持各种代理服务
  - OpenRouter、Azure OpenAI等
  - 标准OpenAI API格式
  - 灵活的模型选择

### 🎯 智能图谱构建
- **实体规范化** - 自动消除歧义和重复
- **关系验证** - 确保图谱完整性和一致性
- **增量处理** - 支持新文档融入现有知识库
- **简体中文优化** - 自动繁简转换和双向链接

### 📝 Obsidian优化输出
- **简洁的文档结构** - 定义和描述分离
- **智能双向链接** - 自动检测和添加`[[实体名称]]`
- **YAML Frontmatter** - 结构化元数据

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/pdf-to-graphmd.git
cd pdf-to-graphmd

# 安装依赖
pip install -r requirements.txt

pip install -e .
```

### API密钥配置

#### Google AI Studio（推荐）
1. 访问 [Google AI Studio](https://aistudio.google.com/)
2. 登录并创建API密钥
3. 设置环境变量：
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

#### OpenAI兼容API
1. 获取代理服务API密钥（如OpenRouter）
2. 设置环境变量：
```bash
export OPENAI_API_KEY="your-openai-api-key"
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

### 准备配置文件

```bash
cp config.yaml.example config.yaml
```

### 配置示例

#### Google原生API配置（推荐）

```yaml
extraction_method: "llm"

# MinerU设置
mineru:
  language: "ch"
  use_gpu: true
  output_formats: ["markdown", "json"]
  batch_size: 1
  vlm_backend: "vlm-transformers"
  source: "modelscope"

# LLM设置 - Google原生API
llm:
  api_provider: "google"  # 使用Google原生API
  google_model_name: "gemini-2.5-pro"
  google_api_key: "your-google-api-key"  # 或通过环境变量设置
  temperature: 0.7
  max_tokens: 4000
  force_json: true
  system_prompt: "You are an expert knowledge extraction system."

# 本体定义
ontology:
  entity_types:
    - "Person"
    - "Organization" 
    - "Location"
    - "Concept"
    - "Theory"
    - "Method"
    - "Tool"
    - "Event"
    - "Document"
    - "Dataset"
  relation_types:
    - "defined_as"
    - "part_of"
    - "related_to"
    - "proposed_by"
    - "used_in"
    - "applies_to"
    - "causes"
    - "results_in"
    - "depends_on"
    - "extends"

# 输出设置
output:
  output_dir: "./obsidian_vault"
  file_extension: ".md"
  include_yaml_frontmatter: true
  include_images: true
  include_tables: true
  include_formulas: true
  language: "chs"  # 简体中文输出
```

#### OpenAI兼容API配置

```yaml
extraction_method: "llm"

# LLM设置 - OpenAI兼容API
llm:
  api_provider: "openai"  # 使用OpenAI兼容API
  model_name: "gemini-2.5-pro"
  api_key: "your-openai-api-key"
  base_url: "https://openrouter.ai/api/v1"
  temperature: 0.7
  max_tokens: 4000
  force_json: true

# 其他配置相同...
```

### API提供商对比

| 特性 | Google原生API | OpenAI兼容API |
|------|---------------|---------------|
| 延迟 | 更低 | 较高（多一层代理） |
| 稳定性 | 更高 | 依赖代理服务 |
| 功能完整性 | 完整 | 可能有限制 |
| 成本 | 直接计费 | 代理服务费用 |
| 配置复杂度 | 简单 | 需要代理设置 |

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
- **MinerU** - PDF解析引擎
- **OpenAI** - OpenAI兼容API支持
- **Google AI** (google-generativeai) - Google原生API支持
- **spaCy** - NLP处理
- **PyTorch** - 深度学习支持
- **OpenCC** - 繁简体中文转换

### 可选依赖
- **CUDA** - GPU加速（推荐）
- **自定义NER模型** - 领域特定实体识别

## 输出示例

生成的Obsidian笔记具有以下结构：

```markdown
# 机器学习

## 定义
一种通过数据训练算法以自动改进性能的人工智能方法。

## 描述
机器学习是[[人工智能]]的一个重要分支，它使计算机能够在没有明确编程的情况下学习。
该领域包含多种算法，如[[监督学习]]、[[无监督学习]]和[[强化学习]]。
[[深度学习]]是机器学习的一个子集，使用[[神经网络]]进行复杂模式识别。

机器学习在[[自然语言处理]]、[[计算机视觉]]等领域有广泛应用。
```

## 许可证

MIT License
