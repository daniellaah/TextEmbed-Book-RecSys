# 开发文档（v0.4）

最后更新: `2026-03-04`

## 1. 项目目标与成功标准

本项目用于量化评估不同文本嵌入模型在推荐系统下游任务中的效果。

当前主任务: `Item-to-Item Retrieval`（语义检索）。

核心问题:

- 在同一数据、同一评估协议下，不同 embedding 模型的检索质量谁更好。

评估成功标准（当前版本）:

- 必须报告: `Recall@10`, `Recall@50`, `NDCG@10`, `NDCG@50`, `MRR@10`, `MRR@50`。
- 必须固定: 数据切分、候选库、索引类型、检索 topK、随机种子。
- 每次实验必须输出可追溯的配置与结果文件（见第 9 节）。

## 2. 技术栈与环境标准

- Python: `3.10`
- 包管理: `uv`
- 深度学习: `PyTorch`
- 模型加载: `transformers`
- 检索: `faiss-cpu`

标准初始化:

```bash
uv python install 3.10
uv venv --python 3.10
uv sync --extra dev
```

建议设置 `uv` 本地缓存目录（便于权限控制与复现）:

```bash
export UV_CACHE_DIR=.uv-cache
```

可复现运行约定:

- 随机种子统一使用: `42`
- 默认设备: `mps`（后续可扩展 `cuda`）
- 日志与产物目录必须写入 `outputs/` 或 `reports/`

## 3. 目录与产物规范

当前仓库将采用以下目录约定（不存在时按需创建）:

```text
data/
  raw/
  processed/
src/
  data/
  embedding/
  retrieval/
  eval/
configs/
  experiments/
reports/
outputs/
docs/
```

各阶段输出:

- `data/processed/items.jsonl`: Item 结构化语料（唯一 item 粒度，不含 `text`）
- `data/processed/interactions.jsonl`: 清洗后的全量交互（保留 rating）
- `data/processed/eval.jsonl`: 评估 query/target 集
- `reports/data_profile/build_items_report.json`: items 构建统计
- `reports/data_profile/build_interactions_report.json`: interactions 构建统计
- `reports/data_profile/build_eval_report.json`: eval 构建统计
- `outputs/embeddings/<model_name>/<experiment_id>/<run_id>/item_embeddings.npy`: 按实验配置产出的 item 向量
- `outputs/index/<model_name>/<experiment_id>/<run_id>/faiss.index`: 按实验配置构建的检索索引
- `outputs/eval/<eval_run_id>/predictions.jsonl`: 每条 query 的 topK 检索结果
- `outputs/eval/<eval_run_id>/run_eval_report.json`: 指标结果
- `outputs/eval/<eval_run_id>/info.json`: 本次评估输入与参数快照

## 4. 数据源与字段约定

数据源（Amazon Reviews 2023 Books）:

- `data/raw/meta_Books.jsonl`
- `data/raw/Books.jsonl`

字段:

- Item 主键: `parent_asin`
- 文本字段: `title`, `description`, `features`, `categories`, `subtitle`, `author`
- 交互字段: `user_id`, `parent_asin`, `rating`, `timestamp`

数据质量规则:

- 去重: item 主键去重（保留信息更完整的一条）。
- 缺失: 关键字段缺失则丢弃（规则需记录在报告）。
- 文本清洗: 去首尾空格，合并连续空白，统一换行为空格。

## 5. Item 字段构造规范

输入:

- `data/raw/meta_Books.jsonl`

输出:

- `data/processed/items.jsonl`

`items.jsonl` 目标 schema:

```json
{"item_id":"B000XXXX","title":"...","subtitle":"...","author":"...","description":"...","features":"...","categories":"..."}
```

字段说明:

- `item_id`: 统一主键，后续模块全部使用该键。
- `items.jsonl` 仅保留结构化字段，不直接落 `text`。
- 文本拼接（view/template）与 embedding 生成由下游阶段负责（见第 7 节）。
- 字段清洗与列表渲染规则由 `build_items.py` 统一实现并固化。

`build_items.py` 清洗与标准化规则:

- 字符串字段（`title/subtitle/author/description`）: `strip` + 连续空白折叠 + 换行替换为空格。
- 原始列表字段先做元素清洗: 丢弃 `null`、空字符串与全空白元素。
- `features` 在 `build_items.py` 中按 `"; "` 渲染为字符串并落盘到 `items.jsonl`。
- `categories` 在 `build_items.py` 中按 `" > "` 渲染为字符串并落盘到 `items.jsonl`。
- 若列表为空，落盘为空字符串 `""`。

脚本路径约定:

- `src/data/build_items.py`

验收标准:

- `items.jsonl` 中 `item_id` 唯一率 `100%`
- `items.jsonl` 中关键字段（`item_id`, `title`）非空率需记录在报告
- 输出行数、去重率、缺失率写入 `reports/data_profile/build_items_report.json`

## 6. User 正样本与评估集构造规范（当前实现）

### 6.1 交互清洗（build_interactions）

输入:

- `data/raw/Books.jsonl`
- `data/processed/items.jsonl`

输出:

- `data/processed/interactions.jsonl`
- `reports/data_profile/build_interactions_report.json`

交互 schema（`interactions.jsonl`）:

```json
{"user_id":"UXXX","item_id":"B000XXXX","rating":5.0,"timestamp":1700000000}
```

构造规则:

- 仅保留合法对象行；坏 JSON/非对象行计入报告并跳过。
- 必需字段: `user_id`, `parent_asin`, `rating`, `timestamp`。
- `parent_asin -> item_id`，且 `item_id` 必须存在于 `items.jsonl`。
- `timestamp` 保留原始量纲（不做秒/毫秒转换）。

脚本路径:

- `src/data/build_interactions.py`

### 6.2 评估集构造（build_eval）

输入:

- `data/processed/interactions.jsonl`

输出:

- `data/processed/eval.jsonl`
- `reports/data_profile/build_eval_report.json`

评估样本 schema（`eval.jsonl`）:

```json
{"user_id":"UXXX","query_item_ids":["B000A","B000B"],"target_item_id":"B000C"}
```

构造规则:

- 正样本定义: `rating >= rating_threshold`（默认 `4.0`）。
- 默认不过滤用户/item 频次（`min_user_pos=1`, `min_item_pos=1`）。
- 当 `min_user_pos > 1` 或 `min_item_pos > 1` 时，启用迭代 k-core 过滤。
- 每用户按 `(timestamp asc, input_order asc)` 排序。
- `target_item_id`: 该用户最后一次正反馈 item。
- `query_item_ids`: 最后一次之前最近 `query_history_n` 次（默认 `1`）。
- 若用户无历史（仅 1 条正反馈），该用户不产出 eval 样本（记录在报告中）。

CLI 关键参数:

- `--rating-threshold`（默认 `4.0`）
- `--min-user-pos`（默认 `1`）
- `--min-item-pos`（默认 `1`）
- `--query-history-n`（默认 `1`）
- `--seed`（默认 `42`）

脚本路径:

- `src/data/build_eval.py`

## 7. Embedding 生成与索引构建

模型对比清单（首批）:

- `BAAI/bge-m3`
- `sentence-transformers/all-MiniLM-L6-v2`
- `intfloat/e5-base-v2`

统一推理参数（必须一致）:

- `embedding_dim`（由实验配置 `model.embedding_dim` 指定）
- `max_length=512`
- `batch_size=64`（CLI 参数，按设备可调整）
- `normalize_embeddings=True`

Embedding 输入约定:

- 由 embedding 阶段基于 `items.jsonl` + 实验配置动态渲染文本，不产出 `items_text.jsonl` 中间文件
- 实验配置统一放在 `configs/experiments/*.yaml`
- `model.name` 必须是 Hugging Face repo id（`namespace/model`），例如 `BAAI/bge-m3`
- embedding 模型只从本机 `~/.cache/huggingface/hub` 读取，不走远程下载
- 对需要自定义模型代码的仓库，可在配置中显式设置 `model.trust_remote_code: true`
- 每个 `view_id` 单独生成一份 item embedding
- 对多视图 embedding 做融合后再建索引，融合方式由 `fusion.method` 指定
- 输入字段清洗与列表渲染规则以第 5 节 `build_items.py` 产物为准。

实验配置模板（示例）:

```yaml
experiment_id: exp_bge_mview_weighted_v1
model:
  name: BAAI/bge-m3
  embedding_dim: 1024
  max_length: 512
  normalize_embeddings: true

text_views:
  views:
    - view_id: view_title
      fields: [title, subtitle, author]
      template: |
        Title: {title}
        Subtitle: {subtitle}
        Author: {author}
    - view_id: view_description
      fields: [description]
      template: |
        Description: {description}
    - view_id: view_features
      fields: [features]
      template: |
        Features: {features}
    - view_id: view_categories
      fields: [categories]
      template: |
        Categories: {categories}

fusion:
  method: weighted_mean
  input_views: [view_title, view_description, view_features, view_categories]
  weights:
    view_title: 0.4
    view_description: 0.2
    view_features: 0.2
    view_categories: 0.2
  normalization: true
```

融合策略（首批）:

- `identity`: 不做多视图融合，直接使用单 view 向量（baseline concat 使用该方法）
- `weighted_mean`: 多 view 加权平均（默认 `view_title=0.4`, `view_description=0.2`, `view_features=0.2`, `view_categories=0.2`）
- 其余融合策略暂不纳入当前版本。

命名约定（避免歧义）:

- `fusion.method` 的 canonical 枚举固定为: `identity`, `weighted_mean`。

`fusion.normalization` 语义:

- 布尔值字段，仅控制“融合后向量是否做归一化”。
- 模型输出 embedding 默认已归一化，不通过该字段控制。
- `identity` 融合下建议设为 `false`（或由实现忽略该字段）。

索引规范:

- 支持 `flat` 与 `hnsw` 两种索引类型（见 `run_eval.py` 参数）。
- 向量统一做 L2 归一化后以内积检索。
- 大规模全量评估建议使用 `hnsw`；`flat` 适合小规模精确验证。

脚本路径约定:

- `src/embedding/generate_item_embeddings.py`
- `src/retrieval/ann_utils.py`
- `src/retrieval/review_item_neighbors.py`
- `src/eval/run_eval.py`

## 8. 评估协议

离线评估指标:

- `Recall@10`, `Recall@50`
- `NDCG@10`, `NDCG@50`
- `MRR@10`, `MRR@50`

评估口径:

- 候选集合: `items.jsonl` 全量 item（可在报告中附加采样实验）。
- 命中定义: ground-truth item 出现在 topK 结果中。
- 显著性: 主报告先给点估计；如需论文级别结论，再补 bootstrap 置信区间。
- query 向量构造: 对 `query_item_ids` 对应 item embedding 做均值，再做 L2 归一化。
- 检索结果会排除 `query_item_ids` 自身，避免历史泄漏。

`run_eval.py` 关键参数:

- `--eval-input`（默认 `data/processed/eval.jsonl`）
- `--embedding-dir`（必填，目录下必须包含 `item_embeddings.npy` 与 `item_ids.jsonl`）
- `--output-root`（默认 `outputs/eval`）
- `--eval-run-id`（可选；不传时使用本机时间 `YYYYMMDDHHMMSS`）
- `--max-query`（默认 `0`，表示不限制；`>0` 表示仅评估前 N 条有效 query）
- `--topk`（默认 `10,50`）
- `--index-type`（`flat`/`hnsw`，默认 `flat`）

评估脚本路径约定:

- `src/eval/run_eval.py`

## 9. 实验记录与可复现要求

每次运行必须生成唯一 `run_id` / `eval_run_id`，并保存:

- embedding 阶段:
  - `outputs/runs/<run_id>/config.json`: embedding 完整配置快照
  - `outputs/embeddings/<model_name>/<experiment_id>/<run_id>/item_embeddings.npy`
  - `outputs/embeddings/<model_name>/<experiment_id>/<run_id>/item_ids.jsonl`
- eval 阶段:
  - `outputs/eval/<eval_run_id>/predictions.jsonl`
  - `outputs/eval/<eval_run_id>/run_eval_report.json`
  - `outputs/eval/<eval_run_id>/info.json`

embedding `config.json` 至少包含:

- 数据版本（文件名 + 修改时间）
- 模型名与 revision
- 推理参数（batch、embedding_dim、max_length、normalize）
- 索引参数（类型、topK）
- 随机种子、设备信息
- 实验配置信息（`experiment_id`, `experiment_config_path`, `view_ids`, `fusion_method`）
- 融合归一化开关（`fusion_normalization`）
- 配置哈希信息（`config_hash`）
- 本次运行产物路径（embedding/index 路径，需包含 `run_id`）

## 10. 开发与 Code/Function Review 规范

分支与提交:

- 分支建议: `feature/*`, `fix/*`, `exp/*`
- commit 信息建议遵循 Conventional Commits

PR 最小检查项:

1. 是否附带输入/输出说明与运行命令。
2. 是否包含最小可复现样例或测试。
3. 是否更新了文档（如改动了数据/评估口径）。
4. 是否给出本次变更风险与回滚方式。

Review 重点:

- 行为正确性: 指标计算、数据切分、主键对齐是否正确。
- 一致性: 不同模型是否严格使用同一评估协议。
- 可复现性: 是否能用同一命令在新环境复跑。
- 性能风险: 大规模数据是否有明显内存/耗时问题。

## 11. 当前待办（按优先级）

P0:

1. 评估结果汇总脚本（读取 `outputs/eval/*/info.json` 与 `run_eval_report.json`）并产出对比表。
2. 固化 baseline 实验配置与运行命令，保证同配置可复跑同结果。
3. 补充失败样本分析模板（基于 `predictions.jsonl`）。

P1:

1. 增加多模型对比实验并汇总结果表。
2. 增加失败样本分析（case study）。
3. 评估 ANN 索引（如 IVF/HNSW）与精度-性能折中。
