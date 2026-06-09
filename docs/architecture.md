# 天枢 架构 / 重构计划

> 本文件记录重定位后的方向、范围、模块边界与里程碑。

> 实现级的详细设计（数据模型 / HTTP 契约 / 路由算法 / 凭据 / serving 流程）见 [design.md](design.md)。

## 重定位（2026-06-08）

天枢从「**云端 provider 桌面客户端**」（登录云账号、注册模型到市场、维持反向隧道卖 GPU）
彻底重定位为「**本地优先的开源 LLM 工具**」，只做两件事：

1. **Local host gateway** —— 本机 OpenAI 兼容网关（聚合 key / 路由 / 回退）。
2. **一键 setup local LLM serving** —— 本机 GPU 一键起 vLLM / llama.cpp。

**已剔除**：云端 REST 客户端、JWT 登录、市场 backend 增删改查、反向 wss/ssh 隧道、账务。

## 仓库

- Repo：`git@github.com:AlisaLi0/tianshu-gateway.git`（独立开源，与原闭源 `llm-gateway` 单体仓分离）。
- 不再依赖 `tianshu-gateway.cloud`；本项目完全本地运行。

## 代码布局

```
tianshu-gateway/
├── Cargo.toml          workspace = [core, cli]（src-tauri 待加）
├── README.md
├── docs/architecture.md
├── core/               Rust 核心库（GUI / CLI 共用）
│   └── src/
│       ├── lib.rs
│       ├── gateway.rs   ← 【新】本地 OpenAI 兼容网关 server（axum）
│       ├── providers.rs ← 【新】上游 provider 注册表（OpenAI/Anthropic/本地兼容）
│       ├── router.rs    ← 【新】model → provider 路由 + 回退
│       ├── serving.rs   ← 【新】一键 serving 编排（argv 组合 + serve_model）
│       ├── gpu.rs       ← 【新】GPU 探测（nvidia-smi/rocm-smi）
│       ├── engine.rs    ← 【重构】本地推理引擎进程管理（vLLM/llama.cpp）
│       ├── models.rs    ← 【留】本地模型仓库 + HF/ModelScope 下载
│       ├── state.rs     ← 【改】配置持久化 + OS 钥匙串（改存 provider key）
│       └── util.rs      ← 【留】
└── cli/                `tianshu` headless CLI
    └── src/main.rs
```

### 模块去留对照（相对旧 client/）

| 模块 | 处理 | 说明 |
|------|------|------|
| `engine.rs` | 留 | 透明 argv 起 vLLM/llama.cpp，spawn/kill/health |
| `models.rs` | 留 | HF/MS 直链下载、磁盘统计 |
| `util.rs` | 留 | tail_file 等 |
| `state.rs` | 改 | 去掉云 gateway URL / JWT；钥匙串改存上游 provider key |
| `gateway.rs` | **重写** | 旧=云端 REST 客户端；新=本地网关 server |
| `tunnel.rs` | **删** | 反向隧道，云端市场专用 |

## 两大功能设计

### 1. Local host gateway（`gateway.rs` + `providers.rs` + `router.rs`）

OpenAI 兼容 server（axum），监听 `127.0.0.1:11435`（可配）：

| 路由 | 行为 |
|------|------|
| `GET /healthz` | 存活探针 |
| `GET /v1/models` | 聚合所有 enabled provider 声明的模型 |
| `POST /v1/chat/completions` | 按 `body.model` 经 `router` 解析出 provider，注入该 provider 的 key 转发上游；流式透传；失败回退到下一候选 |

- **Provider**：`{ name, kind(OpenAI/Anthropic/OpenAICompatible), base_url, api_key_ref, models[], enabled }`。
- **凭据**：`api_key_ref` 指向 OS 钥匙串条目名；明文 key 不进 `providers.json`。
- **路由**：同一 `model` 可由多个 provider 提供，按注册顺序为优先级，连接失败 / 5xx 自动回退。

### 2. 一键 setup local LLM serving（`engine.rs` + `models.rs`）

1. 检测 GPU（`rocm-smi --showmeminfo` / `nvidia-smi`）。
2. 选模型（本地已有，或从 HF/MS 下载）。
3. 生成 vLLM / llama.cpp argv 并 `engine.start`。
4. 健康探活通过后，**自动 `providers.add` 一个指向 `http://127.0.0.1:<port>/v1` 的本地 provider**，于是它立刻出现在网关 `/v1/models` 里。

## 里程碑

1. ✅ **M1 骨架**：workspace + core 模块 + CLI 雏形，`cargo build` 通过。
2. ✅ **M2 网关可用**：`/v1/models` 聚合 + `/v1/chat/completions` 转发 + 回退 + 流式透传，可对接真实 OpenAI 兼容上游。
3. ✅ **M3 一键 serving（CLI）**：`tianshu serve-model` 跑通（GPU 检测 → 起 vLLM/llama.cpp → 健康探活 → 自动注册上游 → 开网关）。已在 4090 机端到端验证。
4. ⏳ **M4 桌面 GUI**：加回 `src-tauri/`，图形化管理 provider / 模型 / 引擎 + 托盘。← 当前
5. ⏳ **M5 打包发布**：Win NSIS / mac dmg / Linux AppImage + `tianshu` 单二进制，挂 GitHub Release。

## 设计取舍

- **透明优先**：引擎 argv 完全由上层组合，core 只管进程生命周期，不藏魔法。
- **本地优先**：默认 `127.0.0.1`，不监听公网；要暴露由用户显式配置。
- **零账号**：不登录、不收集、不回传。
