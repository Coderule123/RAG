# 本地 RAG（DP / RAG / config 分层）

- `DP/`：文档预处理（加载、切分、向量化、入库）
- `RAG/`：运行时流程（检索、重排、上下文、Prompt 生成）
- `config/`：配置与日志（参数加载、HF 环境、日志初始化）

## 1. 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# deactivate the virtual environment
# deactivate
```

## 2. 配置文件

统一配置在 `config/config.yaml`，重点参数：

- `paths.data_dir`：知识库目录（默认 `./RAG/assets/data`）。支持按一级子目录归类，DP 会据此给每个 chunk 打 `tag` 标签：`data/ls6/LS6.txt` -> `tag=ls6`；直接位于 `data/` 下的文件 -> `tag=general`
- `paths.index_dir`：向量库根目录（默认 `./RAG/assets/index_store`），其下由 DP 自动生成 `faiss_store/`、`metadata/`（分块元数据 JSON，**目录结构镜像 `data/`**，如 `data/ls6/LS6.txt` -> `metadata/ls6/LS6.json`；条目中不含 `source`）、`doc_hash.json`（增量用文件哈希表）、`chunks.db`
- `paths.models_dir`：模型缓存目录（默认 `./assets/models`）
- `models.embedding_model`：Embedding 模型
- `models.reranker_model`：Reranker 模型
- `retrieval.top_k` / `retrieval.rerank_top_n`：检索与重排参数
- `huggingface.*`：镜像、离线、本地缓存策略

## 3. DP：构建向量库

运行 `DP/document_processor.py` 即可完成：
加载文档（增量时仅读 `doc_hash.json`）-> 语义切分 -> 向量化 -> 写入 FAISS（LangChain）+ 按文档文件名的 `metadata/*` + `doc_hash.json` + sqlite。

```bash
# export HF_ENDPOINT=https://hf-mirror.com
cd /path/to/仓库根
# cd /home/ymrobot/ws/ymbot/ASR_LLM_TTS/chat_assistant
python3 -m RAG.DP.document_processor
```

可选覆盖参数：

```bash
python3 -m RAG.DP.document_processor \
  --data-dir ./RAG/assets/data \
  --index-dir ./RAG/assets/index_store \
  --chunk-size 500 \
  --overlap 80 \
  --embedding-model BAAI/bge-small-zh-v1.5
```

### 3.1 按 metadata 或 source 删除向量

入库后，`index_store/metadata/` 下每个源文件对应一个 JSON，**目录结构镜像 `data/`**（如 `data/ls6/LS6.txt` -> `metadata/ls6/LS6.json`，`data/car_info.json` -> `metadata/car_info.json`）。条目中 `metadata.doc_id` 与 `metadata.chunk_id` 与 FAISS 内 chunk 一致（`doc_id` 来自 `semantic_splitter`：该次建库时进入切分器的 `Document` 顺序下标；多文件同一批入库时可能不是从 0 连续编号，**以当前 JSON 中显示的值为准**）。

使用脚本 `RAG/DP/index_chunk_delete.py`（须与建库使用相同的 `models.embedding_model`）可按「metadata 文件名」或「原始 `source` 路径」删除，并在成功后**重写** `faiss_store/`、`metadata/`、`doc_hash.json`、`chunks.db`（与 `document_processor` 写入结构一致）。

- 删除某个 metadata 文件中的单条（示例：`car_info.json` 中 `doc_id` 为 0 的全部 chunk；若该条被语义切分为多段且只想删其中一段，再加 `--chunk-id`）：

```bash
cd /path/to/仓库根   # 含顶层包目录 RAG/，且已激活 venv
# cd /home/ymrobot/ws/ymbot/ASR_LLM_TTS/chat_assistant

python3 -m RAG.DP.index_chunk_delete \
  --index-dir ./RAG/assets/index_store \
  --metadata car_info.json \
  --doc-id 0
```

- 删除同一 `doc_id` 下指定分块：

```bash
python3 -m RAG.DP.index_chunk_delete \
  --index-dir ./RAG/assets/index_store \
  --metadata car_info.json \
  --doc-id 4 \
  --chunk-id 1
```

- 删除该源文件在索引中的**全部**向量（`metadata` 下对应 JSON 会随重写消失或变空）：

```bash
python3 -m RAG.DP.index_chunk_delete \
  --index-dir ./RAG/assets/index_store \
  --metadata car_info.json \
  --all
```

- 使用规范后的数据路径（与向量条目中 `metadata.source` 及 `doc_hash.json` 键一致）而非 metadata 文件名：

```bash
python3 -m RAG.DP.index_chunk_delete \
  --index-dir ./RAG/assets/index_store \
  --source RAG/assets/data/car_info.json \
  --doc-id 1
```

说明：`--metadata` 既可传镜像 `data/` 结构的相对路径（如 `ls6/LS6.json`），也可只传 basename（如 `LS6.json`）；若 basename 在多个子目录下重名导致命中多个 `source`，脚本会报错并列出候选 `source`，此时请改用相对路径或 `--source` 明确指定。若删除后索引中**不再存在任何向量**，脚本会清空 FAISS 文件与 `doc_hash.json`（写入 `{}`）并清空 `metadata/` 与 `chunks.db`；之后需重新运行 `document_processor` 建库。

## 4. RAG：生成完整 Prompt（不调用 LLM）

运行 `RAG/rag_api.py` 后，会执行：
检索 -> 重排 -> 上下文拼接 -> 生成最终 Prompt。

```bash
# export HF_ENDPOINT=https://hf-mirror.com
cd /path/to/仓库根
# cd /home/ymrobot/ws/ymbot/ASR_LLM_TTS/chat_assistant
```

- 交互模式

```bash
python3 -m RAG.rag_api
```

- 单次测试

```bash
python3 -m RAG.rag_api --query "RAG是什么？"
python3 -m RAG.rag_api --query "RAG是什么？ --张三"
python3 -m RAG.rag_api --query "RAG是什么？"--user-id "张三"
```

输出是 JSON，包含：

- `contexts`：重排后的上下文
- `prompt`：可直接给后续大模型模块使用的完整提示词

## 5. 日志

- `DP` 日志：默认 `./RAG/logs/doc/`
- `RAG` 日志：默认 `./RAG/logs/`

日志格式：

`[时间][级别][文件:行号]: 消息`

## 6. 模块说明

### config

- `config/config_runtime.py`：读取配置
- `config/logger_runtime.py`：日志初始化
- `config/hf_runtime.py`：HF 镜像/缓存/离线环境

### DP

- `DP/document_loader.py`：文档加载
- `DP/semantic_splitter.py`：语义切分
- `DP/embedding_service.py`：Embedding 封装
- `DP/vector_store.py`：LangChain FAISS 持久化，维护 `doc_hash.json` 与 `metadata/`（镜像 `data/` 目录结构）分文件元数据，并提供 `delete_by_metadata_selector` 按 `source` / metadata 路径与 `doc_id`、`chunk_id` 删除后重写落盘
- `DP/index_chunk_delete.py`：删除脚本入口（调用上述能力）
- `DP/document_processor.py`：预处理流程入口

### RAG

- `RAG/retriever_runtime.py`：向量检索
- `RAG/reranker_service.py`：重排
- `RAG/context_builder.py`：上下文拼接
- `RAG/prompt_builder.py`：Prompt 生成
- `RAG/rag_api.py`：运行时流程入口（到 Prompt 为止）

## 7.接口示例
 
- 初始化

```bash
from RAG.rag_api import RAGService

self.rag_client = RAGService()
```

- 响应示例

```bash
rag_result = self.rag_client.query(user_text)

# 传入视觉姓名
rag_result = self.rag_client.query(query=user_text, user_id="张三")

# 传入声纹姓名
rag_result = self.rag_client.query(query=user_text, vision_user_id=vision_user_id, voice_user_id=voice_user_id)

# 主动招呼（无顾客对话时机器人先开口；仅检索 data/active_ask/，专用 prompt，可不传 query）
rag_result = self.rag_client.query(query="", is_active_ask=True)
# 可选传入情境提示，用于检索与语气：rag_result = self.rag_client.query(query="顾客在展车旁驻足", is_active_ask=True)
```