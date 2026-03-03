# 开发文档（v0.2）

最后更新: `2026-03-02`

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

建议设置本地缓存目录（便于权限控制与复现）:

```bash
export UV_CACHE_DIR=.uv-cache
export HF_HOME=.hf-cache
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
  interim/
  processed/
src/
  data/
    text_preprocess/
    sampling/
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
- `data/processed/interactions_train.jsonl`: 训练交互
- `data/processed/interactions_eval.jsonl`: 评估交互
- `outputs/embeddings/<model_name>/<experiment_id>/<run_id>/item_embeddings.npy`: 按实验配置产出的 item 向量
- `outputs/index/<model_name>/<experiment_id>/<run_id>/faiss.index`: 按实验配置构建的检索索引
- `reports/eval/<run_id>.json`: 指标结果
- `reports/eval/<run_id>.md`: 结果解读与结论

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

- `src/data/text_preprocess/build_items.py`

验收标准:

- `items.jsonl` 中 `item_id` 唯一率 `100%`
- `items.jsonl` 中关键字段（`item_id`, `title`）非空率需记录在报告
- 输出行数、去重率、缺失率写入 `reports/data_profile/items_build_report.json`

## 6. User 正样本与评估集构造规范

输入:

- `data/raw/Books.jsonl`
- `data/processed/items.jsonl`

输出:

- `data/processed/interactions_train.jsonl`
- `data/processed/interactions_eval.jsonl`
- `data/processed/eval_queries.jsonl`

交互 schema:

```json
{"user_id":"UXXX","item_id":"B000XXXX","label":1,"timestamp":1700000000}
```

构造规则（v0.2）:

- 正样本定义: `rating >= 4` 视为正反馈（若无 rating 字段，则用行为存在性定义）。
- 过滤: 用户和 item 至少各有 `5` 次正反馈（可调，需记录）。
- 切分: 按时间切分，最后一次正反馈用于 eval，其余用于 train。

评估 query 构造:

- 每个用户取一个 query item（默认最后一次正反馈之前的一次）。
- 目标 item 为该用户最后一次正反馈 item。

## 7. Embedding 生成与索引构建

模型对比清单（首批）:

- `BAAI/bge-m3`
- `sentence-transformers/all-MiniLM-L6-v2`
- `intfloat/e5-base-v2`

统一推理参数（必须一致）:

- `max_length=512`
- `batch_size=64`（按设备可降）
- `normalize_embeddings=True`

Embedding 输入约定:

- 由 embedding 阶段基于 `items.jsonl` + 实验配置动态渲染文本，不产出 `items_text.jsonl` 中间文件
- 实验配置统一放在 `configs/experiments/*.yaml`
- 每个 `view_id` 单独生成一份 item embedding
- 对多视图 embedding 做融合后再建索引，融合方式由 `fusion.method` 指定
- 输入字段清洗与列表渲染规则以第 5 节 `build_items.py` 产物为准。

实验配置模板（示例）:

```yaml
experiment_id: exp_bge_mview_weighted_v1
model:
  name: BAAI/bge-m3
  max_length: 512
  batch_size: 64
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

- 第一阶段统一用 `faiss.IndexFlatIP`
- 使用 L2 归一化后做内积检索
- 检索 `top_k=100`

脚本路径约定:

- `src/embedding/generate_item_embeddings.py`
- `src/retrieval/build_faiss_index.py`
- `src/retrieval/search.py`

## 8. 评估协议

离线评估指标:

- `Recall@10`, `Recall@50`
- `NDCG@10`, `NDCG@50`
- `MRR@10`, `MRR@50`

评估口径:

- 候选集合: `items.jsonl` 全量 item（可在报告中附加采样实验）。
- 命中定义: ground-truth item 出现在 topK 结果中。
- 显著性: 主报告先给点估计；如需论文级别结论，再补 bootstrap 置信区间。

评估脚本路径约定:

- `src/eval/evaluate_retrieval.py`

## 9. 实验记录与可复现要求

每次运行必须生成唯一 `run_id`，并保存:

- `reports/eval/<run_id>.json`: 指标、样本数、运行时长
- `reports/eval/<run_id>.md`: 结论、异常、后续动作
- `outputs/runs/<run_id>/config.json`: 完整配置快照
- `outputs/embeddings/<model_name>/<experiment_id>/<run_id>/item_embeddings.npy`
- `outputs/index/<model_name>/<experiment_id>/<run_id>/faiss.index`

`config.json` 至少包含:

- 数据版本（文件名 + 修改时间）
- 模型名与 revision
- 推理参数（batch、max_length、normalize）
- 索引参数（类型、topK）
- 随机种子、设备信息
- 实验配置信息（`experiment_id`, `experiment_config_path`, `view_ids`, `fusion_method`）
- 融合归一化开关（`fusion_normalization`）
- 配置哈希信息（`text_config_hash`, `fusion_config_hash`, `full_config_hash`）
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

1. 实现 `build_items.py` 并产出结构化 `items.jsonl`（第 5 节）。
2. 实现评估集构造脚本并固定切分规则（第 6 节）。
3. 跑通一个 baseline 模型全流程并输出首份 `run_id` 报告。

P1:

1. 增加多模型对比实验并汇总结果表。
2. 增加失败样本分析（case study）。
3. 评估 ANN 索引（如 IVF/HNSW）与精度-性能折中。
