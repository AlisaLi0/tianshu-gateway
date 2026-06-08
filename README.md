# 天枢 Tianshu

> 一个本地优先（local-first）的开源 LLM 工具，干两件事：
>
> 1. **Local host gateway** —— 在本机跑一个 OpenAI 兼容网关，聚合多家 provider 的 API key、统一端点、按模型路由 + 失败回退。
> 2. **一键 setup local LLM serving** —— 检测本机 GPU，一键拉起 vLLM / llama.cpp，自动把本地模型注册成网关的一个上游。

两件事天然组合：一键起一个本地模型 → 它自动成为本地网关的上游 → 你的应用只对接 `http://127.0.0.1:11435/v1`，就能同时用上本地模型和各家云端 provider，且 key 永不离开本机。

```
        你的应用 / IDE / agent
                │  OpenAI 兼容请求 (http://127.0.0.1:11435/v1)
                ▼
        ┌─────────────────────────────┐
        │   天枢 Local Host Gateway    │  ← 聚合 key / 路由 / 回退 / 用量
        └───────┬───────────┬─────────┘
                │           │
      ┌─────────▼──┐   ┌────▼──────────────┐
      │ 云端 provider│   │ 本地推理引擎       │  ← 一键 setup
      │ OpenAI/…   │   │ vLLM / llama.cpp  │
      └────────────┘   └───────────────────┘
```

## 为什么放本地

- **少一跳延迟**、**key 不出本机**、**断网也能用本地模型兜底**。
- 不需要云服务器、不需要登录、不收集任何数据。
- 纯个人 / 团队内网的「API key 调度器 + 本地模型管家」，云端那套市场 / 账务 / 隧道**不在本项目范围内**。

## 形态

- **桌面应用**（Tauri 2，Windows / macOS / Linux）：图形化管理 provider、模型、引擎，系统托盘常驻。
- **CLI / headless**（`tianshu`）：服务器 / 无头环境用，`tianshu serve` 起网关、`tianshu model` / `tianshu provider` 管理。

二者共用同一个 Rust 核心库 `core/`。

## 两大功能

### 1. Local host gateway

- OpenAI 兼容：`GET /v1/models`、`POST /v1/chat/completions`（流式 / 非流式）。
- **Provider 聚合**：注册多个上游（OpenAI、Anthropic、任意 OpenAI 兼容端点、本地 vLLM/llama.cpp），各自带 key。
- **路由**：按请求里的 `model` 字段映射到对应 provider；同名模型多 provider 时按优先级 + 失败回退。
- **凭据安全**：上游 key 存 OS 钥匙串（Windows DPAPI / macOS Keychain / Linux secret-service），不落明文。

### 2. 一键 setup local LLM serving

- 检测本机 GPU（ROCm `rocm-smi` / CUDA `nvidia-smi`）。
- 选模型 → 生成并拉起 vLLM / llama.cpp 进程（参数透明，可改）。
- 进程管家：start / stop / 日志 / 健康探活。
- 本地模型仓库：HF / ModelScope 直链下载（无需 git），磁盘占用统计。
- 起好的本地引擎自动注册成网关上游。

## 快速开始（CLI）

```bash
# 起本地网关（默认 127.0.0.1:11435）
tianshu serve

# 加一个云端 provider（key 进钥匙串）
tianshu provider add openai --base-url https://api.openai.com/v1 --api-key sk-...

# 一键起本地 vLLM，并注册成上游
tianshu serve-model Qwen/Qwen3-8B --engine vllm --port 8000

# 现在应用直接打 http://127.0.0.1:11435/v1
```

## 状态

早期重构中（从旧的「云端 provider 隧道客户端」重定位而来，云端相关代码已剔除）。详见 [docs/architecture.md](docs/architecture.md)。

## 许可

开源（许可证待定，倾向 MIT / Apache-2.0）。
