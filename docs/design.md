# 天枢 详细设计（Design）

> 本文档描述天枢各组件的实现级设计：数据模型、HTTP 契约、路由算法、凭据处理、
> 一键 serving 流程、磁盘布局、并发与安全模型。与代码一一对应（`core/src/*.rs`）。
> 高层定位与里程碑见 [architecture.md](architecture.md)。

## 1. 设计原则

- **本地优先**：默认只听 `127.0.0.1`，不监听公网；暴露需用户显式配置。
- **零账号**：不登录、不收集、不回传。纯本机工具。
- **透明优先**：推理引擎的 argv 完全由上层组合，core 只管进程生命周期，不藏魔法。
- **密钥隔离**：上游 provider 的 API key 只进 OS 钥匙串，永不落明文磁盘、永不回传给下游调用方。
- **OpenAI 兼容**：对下游暴露标准 `/v1`，对上游也用 `/v1` 转发，最大化生态兼容。

## 2. 组件总览

```
core/src/
├── gateway.rs    本地 OpenAI 兼容 HTTP server（axum）
├── providers.rs  上游 provider 注册表（持久化 + 钥匙串）
├── router.rs     model → provider 候选解析 + 排序
├── serving.rs    一键 serving 编排（runtime/argv 组合 + serve_model）
├── engine.rs     本地引擎进程管理（spawn/kill/health/teardown）
├── gpu.rs        GPU 探测（nvidia-smi / rocm-smi）
├── provision.rs  docker / WSL 探测 + runtime 自动选择
├── models.rs     本地模型仓库 + HF/ModelScope 下载
├── state.rs      配置持久化 + 钥匙串凭据 helper
└── util.rs       tail_file 等
```

数据流（local gateway）：

```
下游应用 ──/v1/chat/completions──▶ gateway.rs
                                      │ body.model
                                      ▼
                                  router.rs ──ordered routes──┐
                                      │                        │
                                      ▼                        │
                                  providers.rs（取 base_url）   │
                                      │ + state.rs（取 key）    │
                                      ▼                        │
                              reqwest 转发上游 ◀───回退下一候选─┘
```

## 3. 数据模型

### 3.1 Provider（`providers.rs`）

持久化在 `providers.json`（**不含密钥**）：

```jsonc
{
  "providers": [
    {
      "name": "openai",                       // 唯一短名，也是钥匙串 user 后缀
      "kind": "openai_compatible",            // openai | local | openai_compatible
      "base_url": "https://api.openai.com/v1",// 含 /v1
      "needs_key": true,                       // 是否注入 Authorization: Bearer
      "models": ["gpt-4o", "gpt-4o-mini"],     // 空数组 = 通配（仅作回退候选）
      "enabled": true
    }
  ]
}
```

- `kind`：
  - `openai` — 原生 OpenAI API；
  - `local` — 本地引擎（vLLM/llama.cpp），默认无鉴权；
  - `openai_compatible` — 其它兼容端点（DeepSeek/SiliconFlow/Together…），默认值。
- `api_key()`：运行时按 `needs_key` 从钥匙串取密钥（见 §6），不在结构体里存明文。

### 3.2 Settings（`state.rs`）

持久化在 `settings.json`（全部非敏感）：

```jsonc
{
  "gateway_host": null,        // 默认 127.0.0.1
  "gateway_port": null,        // 默认 11435
  "models_dir": null,          // 默认 <data_dir>/models
  "logs_dir": null,            // 默认 <data_dir>/logs
  "vllm_exe": null,            // 首次运行自动探测
  "llama_server_exe": null
}
```

### 3.3 EngineConfig / EngineStatus（`engine.rs`）

```rust
EngineConfig { name, kind(Vllm|LlamaCpp|Custom), program, args, cwd, env, host, port }
EngineStatus { name, running, pid, last_started_at, log_path, last_error, healthy }
```

引擎实例以用户提供的 `name` 唯一标识；argv 由上层完整组合，core 不拼参数。

### 3.4 LocalModel / DownloadRequest（`models.rs`）

```rust
LocalModel    { repo: "org/repo", abs_path, size_bytes, file_count }
DownloadRequest { repo_id, revision="main", files[], dest_root, source(HF|MS), token? }
```

## 4. HTTP API 契约（`gateway.rs`）

监听 `http://{gateway_host}:{gateway_port}`，默认 `127.0.0.1:11435`。

| 方法 | 路径 | 行为 |
|------|------|------|
| GET | `/healthz` | 返回 `200 ok`（存活探针） |
| GET | `/v1/models` | 聚合所有 **enabled** provider 的 `models`，去重排序，返回 OpenAI `list` 结构（`owned_by:"tianshu"`） |
| POST | `/v1/chat/completions` | 取 `body.model` → `router::resolve` → 依次转发候选上游；流式透传响应体；上游 5xx 或连接失败则回退下一候选 |

错误信封（统一）：

```json
{ "error": { "message": "...", "type": "tianshu_error" } }
```

- 缺 `model` → `400`。
- 无 enabled provider 服务该 model → `404`。
- 所有候选都失败 → `502`，message 含最后一次失败原因。

**流式透传**：用 `reqwest::Response::bytes_stream()` → `axum::body::Body::from_stream`，透传上游 `Content-Type`（SSE 时为 `text/event-stream`），不缓冲整体。

**回退时机**：只在「尚未开始流式响应体」之前回退——即上游返回了 HTTP 状态但是 5xx，或连接根本没建立。一旦开始流，就不再切换（语义安全，避免半截响应拼接）。

## 5. 路由与回退（`router.rs`）

`resolve(providers, model) -> Vec<Route>`，确定性排序：

1. **精确匹配优先**：`models` 显式包含该 model 的 provider，按注册表顺序排前。
2. **通配兜底**：`models` 为空的 provider（声明"什么都试试"）排后，按注册表顺序。

`Route { provider, upstream_model }`：当前 `upstream_model == 请求 model`；未来可在此处接 per-provider `model_map` 做上游改名（预留扩展点）。

`aggregate_models(providers)`：把所有 provider 的 `models` 并集去重排序，供 `/v1/models`。

> 注册表顺序即优先级——用户调整 `providers.json` 里 provider 的先后即可改优先级。

## 6. 凭据处理（`state.rs` 钥匙串）

- 服务名固定 `KEYRING_SERVICE = "tianshu"`，每个 provider 的 key 以 user = `provider:<name>` 存。
- `save_provider_key / load_provider_key / clear_provider_key`：底层用 `keyring` crate，三平台原生后端：
  - Windows → DPAPI（credential manager）
  - macOS → Keychain
  - Linux → secret-service（dbus）
- `providers.json` 只存 `needs_key: bool`，**不存明文 key**。
- 删除 provider（`Registry::remove`）会连带 `clear_provider_key`，不留孤儿密钥。

## 7. 一键 setup local serving 流程（`serving.rs` + `engine.rs` + `gpu.rs` + `provision.rs` + `providers.rs`）

核心原则：**用户不需要手动装推理引擎**。天枢不自研内核，但负责把引擎准备好——默认拉官方 docker 镜像跑。`Runtime` 决定怎么起：

| runtime | 怎么起 | 适用 |
|---|---|---|
| `native` | 直接 spawn 宿主二进制（PATH / `--program`） | Windows 上捆绑/预编译的 `llama-server.exe` |
| `docker` | `docker run` 官方镜像 | Docker Desktop（Win/mac）或 Linux 原生 docker |
| `wsl-docker` | `wsl [-d <distro>] -- docker run …` | Windows 且 docker 在 WSL2 distro 里 |

`tianshu serve-model <name> <model> --engine vllm --port 8000`（默认 `--runtime auto`：有 docker 就用镜像，否则 native），编排在 `serving::serve_model`：

```
1. 探测 GPU + runtime   gpu::detect + provision::detect_docker（auto 选 docker/wsl-docker/native）
2. 组合启动命令       serving::build_command（按 runtime），全量 argv 先打日志
3. 启动进程           engine::start（日志文件、进程组隔离、kill_on_drop、记 teardown）
4. 健康探活           engine::wait_healthy：先 TCP，再 GET /v1/models 2xx
5. 自动注册上游       providers::local_provider(name, 127.0.0.1, port, [model]) → Registry::upsert
                     ⇒ 立刻出现在网关 /v1/models，下游可直接调用
```

各 runtime 生成的启动命令：
- **native**：`vllm serve <model> --host H --port P --served-model-name <id>` / `llama-server -m <gguf> --host H --port P --alias <id>`。
- **docker**：`docker run --rm --name tianshu-<name> [--gpus all] -p 127.0.0.1:<port>:<内部端口> -v <named-vol>:<cache> [-e HF_TOKEN] <image> <容器内引擎参数>`。
  - 镜像默认：vLLM=`vllm/vllm-openai:latest`（内部 8000），llama.cpp=`ghcr.io/ggml-org/llama.cpp:server-cuda`（内部 8080）。
  - **权重在容器内下载**：vLLM `--model <hf-repo>`；llama.cpp `-hf <hf-repo>`。缓存落在**具名 docker volume**（`tianshu-hf-cache` / `tianshu-llama-cache`）——不用宿主路径映射，Docker Desktop 与 WSL 内都干净跑通。
  - **仅本地发布** `-p 127.0.0.1:<port>:…`（本地优先，不上公网）。
- **wsl-docker**：同 docker，外包 `wsl [-d <distro>] -- docker …`。

引擎进程管理细节（`engine.rs`）：
- **进程组隔离**（unix `process_group(0)`）：kill 时不误伤父进程。
- **`kill_on_drop(true)`**：句柄析构即终止子进程。
- **`teardown`**：stop 时除了杀子进程，还跑 `docker rm -f tianshu-<name>`——因为 SIGKILL 前台 `docker run` 客户端并不会停容器，必须显式清。容器崩溃自退则靠 `try_wait` 懒 reap 反映到状态。
- **懒 reap**：`status/list/health` 时 `try_wait` 同步子进程退出（running=false，非零退出记 last_error）。
- **健康**：TCP 可连 + `/v1/models` 2xx；docker 下探 `127.0.0.1:<port>`（映射到容器内部端口）。

`tianshu setup`（`provision::setup_report`）预检：列 GPU + docker 接入方式（直连 / WSL / 无），告诉用户“开箱即用”还是缺什么。

模型下载细节（`models::download`）：
- HF：`https://huggingface.co/{repo}/resolve/{rev}/{file}`；MS：ModelScope repo API。
- 流式写 `.part` 临时文件 → 完成后原子 `rename`，断点不污染目标。
- 支持 `Authorization: Bearer <token>`（私有/限流模型）。
- 进度回调 `DownloadProgress { downloaded, total, done, error }`。

## 8. 磁盘布局

`data_dir` 默认 = OS data dir / `tianshu`（无则 `./data`）：

```
<data_dir>/
├── settings.json      非敏感配置（§3.2）
├── providers.json     provider 注册表（§3.1，无密钥）
├── logs/
│   └── engine-<name>.log
└── models/            默认模型仓库（可被 settings.models_dir 覆盖）
    └── <org>/<repo>/...
```

密钥不在此处——在 OS 钥匙串。

## 9. CLI 面（`cli/src/main.rs`）

```
tianshu [--data-dir <path>] <cmd>

info                                 显示生效路径/配置
gpu                                  探测本机 GPU（nvidia-smi / rocm-smi）
setup                                预检：GPU + docker/WSL 是否就绪（一键 serving 前提）
serve [--host H] [--port P]          起本地网关
serve-model <name> <model>           一键：起引擎 + 注册 + 开网关（前台，Ctrl-C 停引擎+网关）
        --engine vllm|llama-cpp|custom
        --runtime auto|native|docker|wsl-docker   默认 auto（有 docker 就用镜像，否则 native）
        --port P                     宿主端口（docker 下=容器发布端口）
        [--image IMG]                docker 镜像覆盖（默认按引擎）
        [--gpus all|...]             docker --gpus（默认 all；空串禁用）
        [--wsl-distro NAME]          wsl-docker 用的 distro
        [--container-port P]         容器内部端口（默认 vLLM 8000 / llama.cpp 8080）
        [--cache-volume NAME]        模型缓存的具名 docker volume
        [--hf-token T]               gated/限流 repo（native env + docker -e）
        [--program <exe>]            native：覆盖默认 vllm/llama-server
        [--engine-host H]            探针 host，默认 127.0.0.1
        [--served-id ID]             对网关暴露的 model id（默认从 model 推导）
        [--health-timeout S]         等健康秒数（默认 600）
        [--gateway-host H] [--gateway-port P]
        [-- <额外引擎参数>]         原样追加到引擎命令
provider list                        列出 provider
provider add <name> --base-url U     增/改 provider
        [--kind openai|local|openai-compatible]
        [--api-key K]                （K 进钥匙串，不落盘）
        [--models a,b,c]
provider rm <name>                   删 provider（连带删钥匙串密钥）
provider enable|disable <name>
model list                           列出本地模型
model download <repo> --files a,b    从 HF/MS 下载（--source hf|ms，--token）
model rm <repo>                      删本地模型目录
```

GUI（`app/`，Tauri 2）复用同一 `tianshu-core`，命令对称（见 §9.1）。

### 9.1 桌面 GUI（`app/src/main.rs` + `app/frontend/`）

Tauri 2 薄壳，把 CLI 的操作暴露成 Tauri commands，前端是 **vanilla HTML/CSS/JS**（`withGlobalTauri: true`，直接 `window.__TAURI__.core.invoke`，不引 npm 构建链——契合单二进制/本地优先）。

- **共享状态** `Ctx { state, registry: Arc<Registry>, engines: Arc<Engines>, gateway: Mutex<Option<GatewayRun>> }`，`tauri::manage` 注入。
- **网关托管运行**：`gateway_start` 用 `gateway::serve_until` 在后台 task 起网关，持 `oneshot` shutdown 句柄；`gateway_stop` 发信号优雅停。provider 增删即时生效（共享同一 `Registry`）。
- **commands**：`app_info / gpu_detect / setup_report / provider_{list,add,remove,set_enabled} / gateway_{status,start,stop} / engine_{list,log,stop} / serve_model / model_list`。
- **serve_model 异步**：后台 task 跑 `serving::serve_model`，完成后 `emit("serve-result", …)`；命令立刻返回引擎名，前端先显示“starting…”，靠 event + 轮询刷新引擎表。
- **密钥不出壳**：`Provider` 只带 `needs_key: bool`，命令从不回传明文 key（仍在钥匙串）。
- **系统托盘**：托盘菜单 Show/Quit；左键切换主窗显隐。**关窗 = 隐藏到托盘**（`CloseRequested` → `prevent_close` + hide），真正退出走托盘 Quit。
- **退出清理**：`RunEvent::ExitRequested` → 停网关 + `engines.stop` 每个引擎（docker 容器借此 `docker rm -f`，不留泄漏）。
- **前端**：四个 tab（Dashboard 网关开关+环境预检+路径 / Providers 增删+开关 / Serving 一键起+引擎表+日志弹窗 / Models 本地模型）。

## 10. 并发模型

- 网关：axum + tokio 多任务；每请求一个 handler，`reqwest::Client` 复用连接池（`GatewayState` 内 `Arc`）。
- 引擎：`Engines` 持 `Mutex<HashMap<name, EngineHandle>>`，每实例一个子进程句柄；查询时懒 reap（`try_wait`）同步子进程退出状态。
- 注册表：`Registry` 持 `RwLock<Vec<Provider>>`，读多写少；每次写后整体持久化 `providers.json`。
- 网关 `reqwest::Client` **无全局超时**——chat 流式可能长时间运行，不能被超时切断。⚠️ reqwest 默认 client 本就无超时，**别调 `.timeout()`**；`Duration::from_secs(0)` 不是“无超时”而是“立即超时”。

## 11. 安全模型

- **默认本地**：bind `127.0.0.1`，不经任何公网；要对外暴露是用户显式改 `gateway_host`。
- **密钥不出本机**：上游 key 存钥匙串，转发时才注入 `Authorization`；下游调用方拿不到上游 key（网关不回显）。
- **无鉴权下游**：本地网关默认不校验下游（本机信任模型）；若 bind 到非 loopback，需用户自行加前置鉴权/防火墙（README 会标注）。
- **CSP（GUI）**：Tauri 壳的 webview CSP 仅允许本机来源（M4 落地时配置）。

## 12. 扩展点（预留，未实现）

- `router.rs` 的 `upstream_model`：接 per-provider `model_map` 做上游改名。
- `gateway.rs`：加 `/v1/completions`、`/v1/embeddings` 透传。
- 计量：在转发处累计 token/请求数，落本地 SQLite（仅本机，不回传）。
- 负载均衡：同 model 多 provider 时从"顺序回退"升级为"加权/最少连接"。
