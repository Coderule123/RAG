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

- `paths.data_dir`：知识库目录（默认 `./assets/data`）
- `paths.index_dir`：向量库目录（默认 `./assets/index_store`）
- `paths.models_dir`：模型缓存目录（默认 `./assets/models`）
- `models.embedding_model`：Embedding 模型
- `models.reranker_model`：Reranker 模型
- `retrieval.top_k` / `retrieval.rerank_top_n`：检索与重排参数
- `huggingface.*`：镜像、离线、本地缓存策略

## 3. DP：构建向量库

运行 `DP/document_processor.py` 即可完成：
加载文档 -> 语义切分 -> 向量化 -> 写入 FAISS（LangChain）+ metadata + sqlite。

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ymrobot/ws/ymbot/ASR_LLM_TTS/chat_assistant
python3 -m RAG.DP.document_processor
```

可选覆盖参数：

```bash
python3 -m DP.document_processor \
  --data-dir ./assets/data \
  --index-dir ./assets/index_store \
  --chunk-size 500 \
  --overlap 80 \
  --embedding-model BAAI/bge-small-zh-v1.5
```

## 4. RAG：生成完整 Prompt（不调用 LLM）

运行 `RAG/rag_api.py` 后，会执行：
检索 -> 重排 -> 上下文拼接 -> 生成最终 Prompt。

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ymrobot/ws/ymbot/ASR_LLM_TTS/chat_assistant
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

- `DP` 日志：`./logs/doc/`
- `RAG` 日志：`./logs/`

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
- `DP/vector_store.py`：LangChain FAISS 持久化
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

# 传入用户姓名
rag_result = self.rag_client.query(query=user_text, user_id="张三")
```