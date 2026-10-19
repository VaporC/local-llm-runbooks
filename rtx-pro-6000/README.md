# Deploying Qwen3.8-Flash-Next 125B on NVIDIA RTX PRO 6000 Blackwell (96GB)

> 实战 runbook:在 NVIDIA RTX PRO 6000 Blackwell(96GB)+ Ubuntu 上,用 **vLLM + Docker** 本地跑 Qwen3.8-Flash-Next(125B MoE,激活 6B)。
> **当前推荐方案(2026-10 起):NVIDIA 官方 `Qwen3.8-Flash-Next-FP4-Local` checkpoint + 社区 vLLM 镜像。**
> 实测 **145-195 t/s**、单请求上下文 **262K**、**16 路并发**,工具调用 + 视觉 + 推理分离。所有数字均来自实机验证。
>
> 旧方案(社区 `mazinb` NVFP4 + `vllm/vllm-openai:qwen38-flash-next`,100K / 2 路 / 60-100 t/s)保留在[第十节](#十旧方案mazinb-nvfp4--官方专用镜像),需要特定改版模型时可参考。
>
> 与 AMD/Vulkan 路线([strix-halo-395](../strix-halo-395/))完全不同——本文是 NVIDIA Blackwell 专用方案。

---

## 零、为什么换方案:新旧实测对比

同一台机器、同一张卡,把模型和镜像从社区 NVFP4 换成 NVIDIA 官方 FP4-Local 之后:

| 指标 | 旧:mazinb NVFP4 | **新:NVIDIA FP4-Local** | 变化 |
|---|---|---|---|
| 模型大小 | ~173GB | **~96GB** | 小近一半 |
| 权重占显存 | ~87GB | **71.3GB** | -16GB |
| KV cache 总量 | ~16.2 万 tokens(BF16) | **~87.4 万 tokens(FP8)** | ×5.4 |
| 单请求上下文 | 100K | **262K** | ×2.6 |
| 并发 | 2 路 | **16 路** | ×8 |
| 解码 · 规整任务 | 90-100 t/s | **~195 t/s** | ×2 |
| 解码 · 发散创作 | 50-66 t/s | **~145 t/s** | ×2.2-2.9 |
| 8 路并发聚合 | (最多 2 路) | **618 t/s**,单路仍 ~100 t/s | — |
| Prefill | 未测 | **~1.5 万 t/s**(20.9 万 token 输入,首字 14s) | — |
| 整机系统内存占用 | ~110GB(剩 ~13GB) | **~44GB**(剩 ~80GB) | -66GB |
| 冷启动 | ~25 分钟 | ~31 分钟 | 稍慢 |
| 精度公开数据 | 无 | GPQA 91.5%(BF16 原版 92.2%)、ToolEvalBench 88.7%(原版 89.4%) | 有据可查 |

**为什么快这么多:**
1. **专用 b12x 内核**:MoE / 线性层 / Gated DeltaNet 解码全换成针对 Blackwell 工作站卡(sm_120)调过的内核。
2. **CPU 开销压掉**:激活仅 6B,单步计算量小,"发指令、起内核"的固定开销占大头;完整 CUDA Graph + 异步调度基本消掉了这部分。
3. **每 token 读得更少**:注意力 MXFP8、专家 NVFP4(缩放因子无损压缩)、KV cache FP8。解码受显存带宽限制,读得少就快。
4. **MTP 草稿更准**:草稿头单独优化 + 概率采样,一次猜 3 个 token 接受率更高。

**为什么上下文和并发能放开:** 旧方案权重占满 ~87GB,只剩 ~9GB 给 KV + Mamba 缓存块,只能降并发换上下文。新 checkpoint 权重小 16GB,加上 FP8 KV 每 token 只占一半空间,省下的显存全给了 KV。

> 测试方法说明:数字来自部署后的功能性实测(单次请求、未多轮取平均)。长上下文时解码会慢一些,多路并发时单路速度会下降。不是严格 benchmark。

---

## 一、硬件 / 系统前提

| 项 | 要求 | 验证命令 |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell,96GB,compute cap 12.0(sm_120)。NVFP4 需 Blackwell 硬件 FP4,**Hopper 及更老不行** | `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` |
| 系统内存 | 建议 ≥128GB(PLE n-gram 表卸到内存)。实测整机占用约 44GB(含同机其他服务) | `free -h` |
| 磁盘 | 模型 ~102GB + 镜像 ~31GB。镜像在 Docker 根目录(通常 `/`),注意根分区余量 | `df -h / /data` |
| 系统 | Ubuntu 22.04/24.04,NVIDIA 驱动 ≥ 580 | `nvidia-smi \| head -3` |
| Docker | Docker + NVIDIA 容器运行时 | 见下 |

```bash
docker info | grep -i runtime         # 期望列表里有 nvidia
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # 容器里能看到卡即正常
```
容器里看不到卡:装 `nvidia-container-toolkit`,再 `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。

**显存要独占**:配置用 `--gpu-memory-utilization 0.975`,卡上不能有别的进程。上线前 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认为空。

---

## 二、拉镜像

```bash
nohup docker pull ghcr.io/local-inference-lab/vllm:release-10152026 > pull.log 2>&1 &
```

- 这是 [NVIDIA 模型卡](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) 指定的社区镜像(Local Inference Lab)。该 checkpoint 的 NVFP4 缩放因子用了新的无损压缩格式,**上游 vLLM 和 `vllm/vllm-openai:qwen38-flash-next` 都还不支持**,只能用它。
- **固定这个 tag,别用 latest**:镜像很新、迭代快,固定版本才能保证可复现。
- 约 31GB。国内实测 ghcr.io 直连可拉;`/etc/docker/daemon.json` 里的 `registry-mirrors` 只对 Docker Hub 生效,管不到 ghcr。
- `docker pull` 不断点续传,大层校验/解压时会长时间静默,**那不是卡死**,别用"没网络活动就 kill"的自动化。

---

## 三、下载模型

**国内优先走 ModelScope**(NVIDIA 官方组织 `nv-community` 有同一份):

```bash
pip install -U modelscope
mkdir -p /data/flash-next
nohup modelscope download --model nv-community/Qwen3.8-Flash-Next-FP4-Local \
  --local_dir /data/flash-next/model > /data/flash-next/dl.log 2>&1 &
```
实测 4 个分片并行、每个 2.5-3.2 MB/s,102GB 用了约 1 小时 37 分。

**HF 源的情况(国内):**
- `hf-mirror.com` **没有缓存这个仓库**,请求会 308 跳回 huggingface.co,`hf download` 报 `LocalEntryNotFoundError`。
- huggingface.co 直连能通,但实测只有 ~0.5 MB/s,102GB 要两天以上。
- 海外网络直接 `hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir /data/flash-next/model` 即可。仓库未 gated。

**下完校验(ModelScope 来源建议做):** 用 HF API 拿官方 sha256,逐个比对大文件。实测 23 个 LFS 文件(含 21 个权重分片)全部一致;只有 `.gitattributes` 大小不同,是 ModelScope 改写的仓库元数据,不影响。

```python
# 官方清单:curl -s "https://huggingface.co/api/models/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/main?recursive=true" > tree.json
import json, hashlib, os
base = "/data/flash-next/model"
for f in json.load(open("tree.json")):
    oid = (f.get("lfs") or {}).get("oid")
    if f["type"] != "file" or not oid: continue
    h = hashlib.sha256()
    with open(os.path.join(base, f["path"]), "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""): h.update(chunk)
    print("OK " if h.hexdigest() == oid else "BAD", f["path"])
```

---

## 四、启动

环境变量和 vLLM 参数照抄 NVIDIA 模型卡的 RTX PRO 6000 配置,写进一个启动脚本。本地只改三处:模型走本地目录、API key 从挂载文件读(不出现在 `docker inspect` 里)、按需挂多个模型名。

**`/data/flash-next/launch.sh`:**

```bash
#!/bin/bash
set -e
export B12X_DENSE_SPLITK_TURBO=1
export B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0
export B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1
export B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1
export B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224
export B12X_DYNAMIC_SPLIT_FAST_PREPARE=1
export B12X_DYNAMIC_SPLIT_LOW_SMEM=1
export B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1
export B12X_DYNAMIC_WORK_SOURCE=persistent_grid
export B12X_W4A16_A4_PREFILL=1
export B12X_DISK_BACKEND=io_uring
export B12X_MHC_PDL=1
export B12X_PCIE_ONESHOT_BLOCK_LIMIT=4
export B12X_PCIE_ONESHOT_PDL=1
export B12X_PCIE_ONESHOT_THREADS=512
export CUDA_BINARY_LOADER_THREAD_COUNT=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY
export CUTE_DSL_ARCH=sm_120a
export INSTANTTENSOR_BACKEND=BUFFERED
export NCCL_BUFFSIZE=2097152
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=spcx
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAFETENSORS_FAST_GPU=1
export TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0
export TORCH_CUDA_ARCH_LIST=12.0a
export VLLM_B12X_DENSE_ACTIVATION_MODE=auto
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1
export VLLM_COMPUTE_NANS_IN_LOGITS=0
export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1
export VLLM_LM_HEAD_A16=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_MTP_NVFP4_LM_HEAD=1
export VLLM_MXFP8_LM_HEAD=0
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0
export VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1
export VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_USE_FASTOKENS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_NCCL_USE_COMM_NONBLOCKING=0
export HF_HUB_OFFLINE=1   # 本地模型,不联网

exec vllm serve /model \
    --served-model-name flash-next \
    --api-key "$(cat /run/apikey)" \
    --host 0.0.0.0 --port 8000 \
    --async-scheduling \
    --block-size 64 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --decode-context-parallel-size 1 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gdn-decode-kernel b12x \
    --gpu-memory-utilization 0.975 \
    --kv-cache-dtype fp8 \
    --no-language-model-only \
    --linear-backend b12x \
    --load-format instanttensor \
    --mamba-cache-mode align \
    --mamba-ssm-cache-dtype auto \
    --max-cudagraph-capture-size 64 \
    --max-model-len 262144 \
    --max-num-batched-tokens 6019 \
    --max-num-seqs 16 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-gb 0.0 \
    --moe-backend b12x \
    --pipeline-parallel-size 1 \
    --quantization modelopt_mixed \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","num_speculative_tokens":3}' \
    --tensor-parallel-size 1 \
    --tool-call-parser qwen3_xml
```

**运行容器:**

```bash
cd /data/flash-next
openssl rand -hex 24 > api-key.txt && chmod 600 api-key.txt   # 首次生成 key
chmod +x launch.sh && mkdir -p cache
docker run -d --name flash-next --init \
  --gpus all --ipc host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped -p 8085:8000 \
  -v /data/flash-next/model:/model:ro \
  -v /data/flash-next/api-key.txt:/run/apikey:ro \
  -v /data/flash-next/launch.sh:/launch.sh:ro \
  -v /data/flash-next/cache:/root/.cache \
  --entrypoint /bin/bash \
  ghcr.io/local-inference-lab/vllm:release-10152026 /launch.sh
```

- **平滑换模型时保住老客户端**:`--served-model-name` 可以给多个名字(如 `flash-next old-model-name`),第一个是主名,老配置不改也能连。
- 挂 `/root/.cache` 是为了保留编译缓存,下次启动可能快一些。

**加载进度:**

```bash
docker logs -f flash-next 2>&1 | grep -vE 'rope_parameters|not documented' \
  | grep -E 'Loading weights|Model loading took|KV cache size|Application startup'
```

实测时间线(权重在 SATA 盘上):启动 → 权重加载完约 19 分钟(`Model loading took 71.33 GiB ... 1123s`)→ profile / 编译 / 预热约 12 分钟 → `Application startup complete`,**合计约 31 分钟**。期间显存会在 72-93GB 之间来回跳,属正常。

日志里 `Unrecognized keys in rope_parameters`、`min_frames is part of ... not documented` 带 `[ERROR]` 字样,其实是 transformers 的无害警告,忽略即可。

---

## 五、关键参数说明

| 参数 | 值 | 作用 |
|---|---|---|
| `--quantization` | `modelopt_mixed` | 混合精度:专家 NVFP4、注意力 MXFP8 |
| `--load-format` | `instanttensor` | 该 checkpoint 的加载格式 |
| `--kv-cache-dtype` | `fp8` | KV 每 token 占一半空间,上下文/并发能放开的关键之一 |
| `--gpu-memory-utilization` | `0.975` | 很激进,要求独占显卡 |
| `--max-model-len` | `262144` | 原生 262K(模型卡称可扩到 1M,未验证) |
| `--max-num-seqs` | `16` | 并发上限 |
| `--mamba-cache-mode` | `align` | DeltaNet 状态缓存模式 |
| `--moe-backend` / `--linear-backend` / `--gdn-decode-kernel` | `b12x` | Blackwell 专用内核,提速的主要来源 |
| `--speculative-config` | MTP,3 token,probabilistic | 投机解码 |
| `--tool-call-parser` | `qwen3_xml` + `--enable-auto-tool-choice` | Agent 客户端必需,否则 `tool_choice:auto` 报 400 |
| `--reasoning-parser` | `qwen3` | 思考与正文分离 |
| `VLLM_PLE_CPU_OFFLOAD` | `1` | PLE n-gram 表放系统内存 |

**KV 池怎么用:** 启动日志会打印 `GPU KV cache size: 873,813 tokens, Maximum concurrency for 262,144 tokens per request: 3.33x`。87 万是**所有请求共享的总池**,不是每路 262K:满长度请求最多同时约 3 个,短请求可以开满 16 路。超出 `max-num-seqs` 的请求会**排队**(变慢,不报错)。子 agent 并行时每个各占一路。

---

## 六、实测数据(2026-10-19,单卡 RTX PRO 6000)

| 测试 | 结果 |
|---|---|
| 显存 | 权重 71.3GB,总占用 ~94GB / 96GB |
| KV cache | 873,813 tokens(FP8) |
| 解码 · 规整任务(三张报表说明,159 tokens) | **194.5 t/s**,首字 0.09s |
| 解码 · 发散创作(短篇故事,475 tokens) | **145.2 t/s**,首字 0.08s |
| 8 路并发(每路 300 tokens) | 2.6s 完成,聚合 **618 t/s**,单路平均 101.5 t/s |
| 长文档大海捞针 | 输入 **208,943 tokens**,首字 14.0s(prefill ~14,900 t/s),答案正确 |
| 推理模式 | reasoning_content 与 content 正确分离,财务计算题答对 |
| 工具调用 | `finish_reason=tool_calls`,一次发出 2 个并行调用,参数正确 |
| 看图 | 合成票据的发票号/日期/金额提取正确,0.46s |

规整任务比发散创作快,是因为 MTP 草稿在可预测文本上接受率更高,正常现象。

---

## 七、验证

```bash
K=$(cat /data/flash-next/api-key.txt)
curl -s localhost:8085/health -o /dev/null -w '%{http_code}\n'                     # 200
curl -s localhost:8085/v1/models -H "Authorization: Bearer $K"                       # 列出模型,max_model_len=262144
curl -s localhost:8085/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"北京天气?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":200}'
# 期望 finish_reason=tool_calls
curl -s -o /dev/null -w '%{http_code}\n' localhost:8085/v1/models                     # 无 key 应 401
```

除了 OpenAI 兼容的 `/v1/chat/completions`,这个镜像还提供 `/v1/responses` 和 Anthropic 兼容的 `/v1/messages`。

---

## 八、故障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Qwen4ExpForConditionalGeneration are not supported` | 用了 PyPI vLLM / 普通镜像 | 用本文镜像 |
| 加载 FP4-Local 报量化/缩放因子格式不支持 | 用了上游 vLLM 或旧的 `qwen38-flash-next` 镜像 | 必须用 `ghcr.io/local-inference-lab/vllm:release-10152026` |
| `hf download` 报 `LocalEntryNotFoundError` | hf-mirror 不缓存该仓库,308 跳回 huggingface.co | 改走 ModelScope `nv-community/` |
| 启动时显存不足 / OOM | 卡上有别的进程,0.975 太激进 | 清空其他 GPU 进程;仍不行降到 0.95 并接受 KV 变少 |
| 加载半小时还没好 | 正常,冷启动约 31 分钟 | 看 `Model loading took` 是否已出现;`docker inspect -f '{{.State.Status}}'` 是否 running |
| 日志里 `[ERROR] min_frames ... not documented` | transformers 文档字符串警告 | 无害,忽略 |
| 客户端报 `maximum context length ... total 262145 tokens` 之类 | 客户端把输出 max_tokens 设成"用满剩余上下文",差 1 个 token 就超 | 客户端输出 max_tokens 固定为 8192 |
| 客户端报 `"auto" tool choice requires ...` (400) | 没开工具调用 | 确认 `--enable-auto-tool-choice --tool-call-parser qwen3_xml` |
| 回复混入英文思考 | 没开推理解析器 | 确认 `--reasoning-parser qwen3`;客户端思考模式可关 |
| 客户端报 `Tool 'WebSearch' is not defined` / 不支持原生 WebSearch | 原生 WebSearch 是云厂商服务端功能,本地模型没有 | 客户端里用 MCP 接一个真正的搜索工具(如 SearXNG),模型的工具调用本身没问题 |
| 客户端"连通性测试"按钮转圈(如 Trae) | 客户端 bug,请求没发到服务器 | 忽略按钮,直接保存、发消息验证 |

---

## 九、生产加固

- **自启**:`--restart unless-stopped`,机器重启后 Docker 自动拉起(仍要等 ~31 分钟加载)。
- **API key**:放文件、只读挂进容器,不写在 docker 参数里(否则 `docker inspect` 能看到明文)。换 key 要重启容器,攒到下次重启顺手做。
- **外网访问务必上 HTTPS**:明文 HTTP 下 API key 和业务数据在公网可被截获。在前面加一层反向代理(nginx/Caddy)做 TLS,要点:
  - SSE 流式不能缓冲:nginx `proxy_buffering off;`,否则客户端要等整段生成完才显示
  - 超时放长:`proxy_read_timeout 600s` 以上(长文档请求要跑几分钟)
  - 请求体放大:`client_max_body_size 100m`(图片 base64 很大)
  - 原样转发 `Authorization` 头
  - 配好后**关掉公网上的明文端口**;证书 90 天有效,确认自动续期
- **客户端配置**:输入上下文设 ~250000(留余量),**输出 max_tokens 固定 8192**。
- **回退预案**:换新方案时旧容器先只停不删(`docker update --restart=no` 防止开机抢 GPU),跑稳一两周再清理。

---

## 十、旧方案:mazinb NVFP4 + 官方专用镜像

> 2026-10 之前的生产方案。新方案在速度、上下文、并发、内存上全面更好,**新部署不推荐**。仅当你需要某个只有社区 NVFP4 版本的改版模型时参考。

- 镜像:`vllm/vllm-openai:qwen38-flash-next`(vLLM ≥ 0.29;PyPI vLLM 不认此架构)
- 模型:`mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4`(gated,~173GB)或 `primitive-ai/Qwen3.8-Flash-Next-NVFP4`。下载用 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`(不关 Xet 会绕过镜像报 401)
- 资源:权重占显存 ~87GB,PLE 表占系统内存 ~100GB,冷启动 ~25 分钟

```bash
docker run -d --name flash-next --gpus all --ipc=host -p 8085:8000 --restart unless-stopped \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  -v /data/flash-next/model:/model \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model --served-model-name flash-next --api-key "$APIKEY" \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.93 --max-model-len 100000 --max-num-seqs 2 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

旧方案的特有坑:
- `--distributed-executor-backend mp` **单卡也必须加**,否则 PLE 卸载 worker 不启动
- `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800`:默认 600 秒灌不完 PLE 表
- **并发和上下文抢显存**(只剩 ~9GB):并发 32 / 4 / 2 对应上下文约 24K / 40K / 100K。`max_num_seqs (N) exceeds available Mamba cache blocks (M)` 时把并发降到 ≤ M

---

## 附:速查

- 镜像:`ghcr.io/local-inference-lab/vllm:release-10152026`(固定 tag)
- 模型:`nvidia/Qwen3.8-Flash-Next-FP4-Local`(~96GB;国内走 ModelScope `nv-community/`)
- 配置:照抄 NVIDIA 模型卡,262K 上下文 / 16 路 / FP8 KV / 0.975 显存
- 实测:规整 ~195 t/s、发散 ~145 t/s、8 路聚合 618 t/s、prefill ~1.5 万 t/s
- 资源:显存 ~94GB(权重 71GB),整机内存 ~44GB,冷启动 ~31 分钟
- 客户端:输出 max_tokens 固定 8192;外网务必 HTTPS
