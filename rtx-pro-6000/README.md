# Deploying Qwen3.8-Flash-Next 125B on NVIDIA RTX PRO 6000 Blackwell (96GB)

> 实战 runbook:在 NVIDIA RTX PRO 6000 Blackwell(96GB)+ Ubuntu 上,
> 用 **vLLM + Docker + NVFP4 量化**,本地跑 Qwen3.8-Flash-Next(125B MoE,激活 6B)。
> 实测 **60-100 t/s**(视任务),工具调用 + 视觉 + 单请求上下文最高 100K。所有配置均来自实机验证。
>
> 与 AMD/Vulkan 路线([strix-halo-395](../strix-halo-395/))完全不同——本文是 NVIDIA Blackwell + vLLM + NVFP4 张量核的专用方案。


# 部署 Qwen3.8-Flash-Next 125B(NVIDIA RTX PRO 6000 / Blackwell)

> 目标:全新 Ubuntu 机器 → 本地跑 125B MoE 模型 + MTP 加速 + 视觉 + 工具调用,OpenAI 兼容 API。
> 实测:规整任务 ~90-100 t/s、发散创作 ~50-66 t/s,单请求上下文 24K-100K(可调),显存占用约 88-91GB / 96GB + 约 100GB 系统内存。
> 说明:俗称"135B"实为 **125B 总参数、每 token 激活 6B**(qwen4_exp 超稀疏 MoE)。它带一张 **~51B 参数的 PLE n-gram 查找表**,这是本方案所有特殊之处的根源。

---

## 零、为什么这套方案(先理解,再动手)

三个关键事实,每个都是踩坑换来的,决定了整套流程和普通模型不一样:

1. **架构太新,标准工具全不认**。PyPI 版 vLLM(`pip install vllm`,哪怕 0.28)、普通 `vllm/vllm-openai:latest`、主线 llama.cpp,加载都报 `Qwen4ExpForConditionalGeneration are not supported`。**必须**用官方专用镜像 `vllm/vllm-openai:qwen38-flash-next`(vLLM ≥ 0.29)。
2. **PLE 表要占约 100GB 系统内存**(不是显存)。靠 `VLLM_PLE_CPU_OFFLOAD=1` 卸载到 RAM。所以这台机器除了 96GB 显存,还必须有 128GB 内存、其中 ~100GB 空闲。
3. **并发数和上下文长度直接抢显存**。权重占 ~87GB,只剩 ~9GB 给 KV cache + Mamba(DeltaNet)缓存块。降并发是扩上下文的唯一有效手段(见第七节)——这是最反直觉的地方。

另外:NVFP4 用 Blackwell 的硬件 FP4 张量核,**Hopper(H100/H200)及更老的卡跑不了**;冷启动约 25 分钟(权重 173GB + 双 worker 各读一遍 + 慢盘),是一次性成本。

---

## 一、硬件 / 系统前提(部署前自检)

| 项 | 要求 | 验证命令 |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell,96GB,compute cap 12.0(sm_120) | `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` |
| 系统内存 | ≥128GB,空闲 ≥100GB(给 PLE 表) | `free -h` |
| 磁盘 | ≥200GB 空闲放模型。强烈建议 NVMe(慢盘冷启动极久) | `df -h /data` |
| 系统 | Ubuntu 22.04/24.04,NVIDIA 驱动 ≥ 580(CUDA 13 兼容) | `nvidia-smi \| head -3` |
| Docker | Docker + NVIDIA 容器运行时 | 见下 |

**驱动 + 容器运行时自检(必须能在容器里看到 GPU):**
```bash
nvidia-smi   # 主机能看到卡
docker info | grep -i runtime         # 期望列表里有 nvidia
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # 容器里也能看到卡即正常
```
若容器里看不到卡:装 `nvidia-container-toolkit` 并 `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。

任何一项(尤其内存)不满足**先解决再往下**——PLE 表灌不进内存会在加载后期才失败,白等 20 分钟。

---

## 二、Docker 镜像源(仅国内网络需要)

直连 Docker Hub 拉这个几十 GB 的镜像**极不稳**:会卡在某层无限重试,且 `docker pull` 不断点续传(连接一断,没下完的层作废重来)。配国内镜像:
```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1panel.live"]
}
EOF
sudo systemctl restart docker
```
⚠️ 若机器上已有带 `--restart` 的容器,重启 docker 会把它们一并拉起(甚至 k8s kind 之类会自己复活),重启前心里有数。

---

## 三、拉专用镜像

```bash
docker pull vllm/vllm-openai:qwen38-flash-next
```
**拉取中途别用"检测到没网络活动就 kill 重来"的自动化。** docker 在校验(Verifying Checksum)/解压(Extracting)大层时会有很长的静默期,那**不是卡死**,粗暴 kill 反而让进度归零、每次从头。正确做法:`nohup docker pull ... > pull.log 2>&1 &` 后台拉,隔几分钟看 `docker images | grep vllm-openai` 是否出现即可。真要判断死活,看 `ss -tnp | grep dockerd` 有没有活跃连接。

---

## 四、下载模型(gated 仓库,HF token)

推荐单卡友好的 NVFP4 量化版,如 `mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4`(~173GB)或 `primitive-ai/Qwen3.8-Flash-Next-NVFP4`(带低内存 PLE 表可选)。这类仓库多为 **gated**:先去该模型 HF 网页点"Agree and access"(多为自动通过),再用带权限的 token。

```bash
pip install -U "huggingface_hub[cli]"
echo "hf_xxxxx" > ~/.hf_token && chmod 600 ~/.hf_token
export HF_TOKEN=$(cat ~/.hf_token) HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
mkdir -p /data/flash-next/model
nohup hf download mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
  --local-dir /data/flash-next/model > /data/flash-next/dl.log 2>&1 &
```

**下载坑:**
- **必须 `HF_HUB_DISABLE_XET=1`**。hf 新版默认走 Xet 后端,绕过镜像直连 xethub 在国内报 401。
- gated 仓库匿名/无授权 token 会报 `Access to model ... is restricted`——去网页点同意 + 换授权 token。
- 支持断点续传,中断/`credentials expired` 重跑同一条即可。~173GB 走镜像数小时。
- **视觉能力混在权重分片里,不需要单独 mmproj 文件**(和 llama.cpp/GGUF 约定不同)。

---

## 五、启动容器

先用保守参数跑通,再按需调上下文(第七节)。生产版完整命令:
```bash
APIKEY=$(cat ~/.flash-next-key 2>/dev/null || openssl rand -hex 24 | tee ~/.flash-next-key)
docker rm -f flash-next 2>/dev/null
docker run -d --name flash-next \
  --gpus all --ipc=host \
  -p 8085:8000 \
  --restart unless-stopped \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  -v /data/flash-next/model:/model \
  -v /data/flash-next/hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model \
  --served-model-name flash-next \
  --api-key $APIKEY \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.90 \
  --max-model-len 24576 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

四个**不能省**的点(省了会以各种形式报错,全在故障表):
- `VLLM_PLE_CPU_OFFLOAD=1` + `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800`:PLE 表卸载到内存;默认 600 秒 timeout 灌不完 95GB 表会超时,设 1800。
- `--distributed-executor-backend mp`:**单卡也必须加**,否则 PLE 卸载 worker 根本不启动。
- `--enable-auto-tool-choice --tool-call-parser qwen3_xml`:Agent 客户端(Trae/Cline 等)默认发 `tool_choice:auto`,不加报 400。
- `--reasoning-parser qwen3`:分离思考与正文,否则模型的英文思考过程泄露进回复。

---

## 六、关键参数详解(勿乱改,每个都验证过)

| 参数 | 值 | 作用 / 为什么 |
|---|---|---|
| `--gpus all --ipc=host` | — | 容器可见 GPU;`--ipc=host` 供 vLLM 共享内存,不加可能报 shm 错 |
| `-e VLLM_PLE_CPU_OFFLOAD=1` | — | 把 ~51B PLE 表卸到系统内存(~100GB RAM),不加显存装不下 |
| `-e VLLM_PLE_OFFLOAD_READY_TIMEOUT` | `1800` | PLE worker 就绪超时(秒),默认 600 不够 |
| `--distributed-executor-backend` | `mp` | **单卡也必须**,否则 PLE 卸载 worker 不启动 |
| `--gpu-memory-utilization` | `0.90~0.93` | 权重已占 ~87GB;0.93 再高有 OOM 风险(留运行时+碎片余量) |
| `--max-model-len` | 见第七节 | 单请求上下文上限,受 KV cache 显存限制 |
| `--max-num-seqs` | 见第七节 | = Mamba 缓存块数;和上下文抢显存,主调节手段 |
| `--tool-call-parser` | `qwen3_xml` | Qwen3.8 系列专用;名字错了 vLLM 启动会立刻报 invalid choice 并列出可选值 |
| `--reasoning-parser` | `qwen3` | 分离 reasoning_content 与 content |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | MTP 投机解码加速;用 3(1/2 在某些版本 boot hang) |
| `--restart unless-stopped` | — | Docker daemon 开机自启,等价服务开机自启 |

**速度预期:**规整任务(问答/翻译/结构化)MTP 接受率高(~0.7),~90-100 t/s;发散创作接受率低(~0.3-0.4),~50-66 t/s。都正常,MTP 收益本就取决于任务可预测性。

---

## 七、上下文与并发的权衡(实测边界,极重要)

**核心事实:** 96GB 显存,权重占 ~87GB,只剩约 9GB 给 KV cache + Mamba 缓存块。并发和上下文抢同一块显存,**降并发换上下文是唯一有效手段**。

实测数据表(gpu-memory-utilization 0.90-0.93):

| `--max-num-seqs`(并发) | KV cache 可用 | KV 池总量 | 单请求上下文上限 |
|---|---|---|---|
| 32 | ~1.6 GiB | ~2.5K | 24K(勉强) |
| 4 | ~4.6 GiB | ~98K | 40K(舒适) |
| 2 | ~5.6 GiB | ~162K | **100K** |

从 32→2,上下文翻了 4 倍。**怎么定值:别猜,让 vLLM 算**——`--max-model-len` 先设偏大(如 100000),启动时它会报:
```
GPU KV cache size: 161,904 tokens, Maximum concurrency for 100,000 tokens per request: 1.62x
```
或 `ValueError: ... estimated maximum model length is XXXXX`。照它给的实际可用值(留 5-10% 余量)重设、重启定死。

**并发要设多少 —— 关键:子 agent 并行时每个占一路。** 主 agent 发 1 个 + 同时派 3 个子 agent = 那一刻 4 路。超出 `max-num-seqs` **不报错,而是排队**(变慢,非失败)。所以:
- 单线程长对话/长文档、少并行 → 并发 2,换 100K 上下文
- Agent 编排、常并行派子任务 → 并发 4(40K),避免并行请求互相排队卡顿

**改配置的代价:** 每次改 `max-num-seqs`/`max-model-len` 都要 `docker rm -f` + 重跑 = 一次 ~25 分钟完整重新加载。想试不同档位,**攒一起一次调好**,定下最终值写进部署文档。

---

## 八、验证(部署完逐项过)

```bash
K=$(cat ~/.flash-next-key)
# 1) 模型列表
curl -s localhost:8085/v1/models -H "Authorization: Bearer $K"
# 2) 推理
curl -s localhost:8085/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
# 3) 工具调用(决定能不能给 Agent 用,必测;期望 finish_reason=tool_calls 且参数抽对)
curl -s localhost:8085/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"北京天气?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":200}'
# 4) 无 key 应 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8085/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"x"}]}'
```
加载进度看:`docker logs -f flash-next 2>&1 | grep -E 'Completed \||KV cache|Application startup'`,看到 `Application startup complete` 即就绪。

---

## 九、故障速查(踩过的坑合集)

| 症状 | 原因 | 处理 |
|---|---|---|
| `Qwen4ExpForConditionalGeneration are not supported` | 用了 PyPI vLLM / 普通镜像 | 必须用 `vllm/vllm-openai:qwen38-flash-next`(≥0.29) |
| docker pull 卡住 / 反复重试某层 | 直连 Docker Hub 不稳,且不断点续传 | 配 registry mirror(第二节);静默期≠卡死,别乱 kill |
| 下载 `401` / `Access ... restricted` | gated 未授权,或走了 Xet | HF 网页点同意 + 授权 token;加 `HF_HUB_DISABLE_XET=1` |
| 加载后期报 KV cache 不足(`estimated maximum model length is X`) | `max-model-len` 太大 | 降到 X 以内,或降 `max-num-seqs` 换 KV 显存(第七节) |
| `max_num_seqs (N) exceeds available Mamba cache blocks (M)` | DeltaNet 每并发占一个 Mamba 块,显存只够 M 个 | `max-num-seqs` 降到 ≤ M(留余量) |
| PLE 表加载超时 / offload worker 起不来 | 缺 `--distributed-executor-backend mp` 或缺 timeout 或内存不足 | 补齐两者;`free -h` 确认 ≥100GB 空闲 |
| 加载"卡住"、显存/内存数字长时间不变 | 多半不是卡死,是双 worker 先后跑+慢盘 | 看 `grep 'Completed \|'` 分片百分比是否在涨;`docker inspect -f '{{.State.Status}}'` 是否 running |
| 客户端报 `"auto" tool choice requires --enable-auto-tool-choice...` (400) | 服务端没开工具调用 | 加 `--enable-auto-tool-choice --tool-call-parser qwen3_xml` |
| 回复混入英文自言自语的思考 | 没开推理解析器 | 加 `--reasoning-parser qwen3`;客户端思考模式设关闭 |
| 客户端"连通性测试"按钮转圈(如 Trae) | 客户端 bug,请求根本没发到服务器 | 忽略测试按钮,直接保存、发实际消息;服务端日志无对应 POST 即可佐证 |
| 请求报 `maximum context length ... total YYYYY tokens` | 输入+输出超过 max-model-len | 客户端把输出 max_tokens 设小(4K)、输入设接近上限;或开新对话清历史;或第七节降并发扩上下文 |
| HTTP 000(连不上,非 401/400) | 网络层没通(不是鉴权) | 查 ping/端口/中间防火墙;区别于 401(通了但没权限) |

---

## 十、可选加固(生产建议)

- **开机自启**:`--restart unless-stopped` 已让容器随 Docker 自动拉起,机器重启/崩溃不用手动救(省下 25 分钟)。
- **API key**:`--api-key` 是唯一门槛。**公网暴露前必须换掉调试期用过的 key**;换 key 要重启容器(25 分钟),攒到下次重启顺手做。
- **加载提速**:把模型放 NVMe(SATA 慢盘是 25 分钟的主因);根分区不够先腾(常见大户 `~/.cache`)。
- **前端 / Agent**:Trae/Cline 等填 `http://<IP>:8085/v1` + 模型名 `flash-next` + key;输出 max_tokens 设小、输入贴合服务端 max-model-len。
- **共享机注意**:若这台卡还跑别的服务,Flash-Next 需要近乎独占显存(88GB)+ 100GB 内存;上线前用 `nvidia-smi` / `free -h` 确认余量,必要时先停别的服务腾资源(记得记录以便恢复)。
- **别对公网裸暴露**:内网 IP 机器要公网访问,放行动作在网关/路由器的端口映射上,不在这台 Linux;暴露前务必换强 key、并评估这是一台 uncensored 模型的风险。

---

## 附:速查——一句话记住每个数字

- 镜像:`vllm/vllm-openai:qwen38-flash-next`(≥0.29,PyPI vLLM 不认此架构)
- 模型:`mazinb/...-NVFP4`(gated,~173GB,视觉在权重里无需单独 mmproj)
- 必需环境变量:`VLLM_PLE_CPU_OFFLOAD=1` + `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800`
- 必需参数:`--distributed-executor-backend mp`(单卡也要)+ `--tool-call-parser qwen3_xml` + `--reasoning-parser qwen3` + MTP spec
- 上下文/并发:并发 32/4/2 ↔ 上下文 24K/40K/100K,让 vLLM 报数别猜;子 agent 并行各占一路
- 资源:显存 ~88-91GB / 96GB,系统内存 ~100GB(PLE 表),冷启动 ~25 分钟(一次性)
- 实测:规整 ~90-100 t/s,发散 ~50-66 t/s,工具调用✅ 视觉✅
