# Deploying Qwen3.8-Flash-Next 125B on AMD Ryzen AI Max+ 395 (Strix Halo)

> 实战 runbook:在 AMD Ryzen AI Max+ 395(Strix Halo / gfx1151 / Radeon 8060S,128GB 统一内存)+ Ubuntu 上,
> 用 **Vulkan + 自编译 MTP fork 引擎**,本地跑 Qwen3.8-Flash-Next(125B MoE,激活 6B)。
> 实测散文/代码 **29-40 t/s**,MTP 接受率 **0.8-0.94**,视觉 + 256K 上下文。所有配置均来自实机验证。
>
> 与 NVIDIA/vLLM 路线完全不同——本文是 AMD 核显 + Vulkan + ROCmFP4 的专用方案。


# 部署 Qwen3.8-Flash-Next 125B(AMD AI Max+ 395 / Strix Halo)

> 目标:全新 Ubuntu 机器 → 本地跑 125B MoE 模型 + MTP 加速 + 视觉,OpenAI 兼容 API。
> 实测:散文/代码 **29-40 t/s**,MTP 草稿接受率 **0.80-0.94**,单请求上下文 **192K**(可选 256K),显存占用约 75GB / 112GB。
> 说明:模型俗称"135B"实为 **125B 总参数、每 token 激活 6B**(MoE 稀疏)——这是它能在核显上跑到 30+ t/s 的根本原因。

---

## 零、为什么这套方案(先理解,再动手)

三个关键选型,每一个都是踩坑换来的,别自作主张换掉:

1. **后端用 Vulkan/RADV,不用 ROCm**。同硬件实测:Vulkan 生成速度 ~80 t/s vs ROCm ~59 t/s(ROCm 只在 prefill 略快)。生成速度是交互体验的决定指标,Vulkan 赢。
2. **引擎必须用特定 fork,不能用官方主线 llama.cpp**。Flash-Next 的 MTP 头官方主线还不支持,直接加载会崩 `invalid vector subscript`。只有 `LaurentZuijdwijk/llama.cpp` 的 `vulkan/qwen4exp-rocmfpx` 分支实现了 qwen4exp 架构 + ROCmFP4 张量 + MTP 图,且专门为 Strix Halo 调优 Vulkan。
3. **模型用 STRIX_LEAN 专用量化 + 配套 MTP 草稿头**,不用通用 Q4。整套(fork 引擎 + STRIX_LEAN 主模型 + agentionai 草稿头)是绑定验证的组合,混用通用量化引擎认不出张量布局。

---

## 一、硬件 / 系统前提(部署前自检)

| 项 | 要求 | 验证命令 |
|---|---|---|
| CPU/GPU | AMD Ryzen AI Max+ 395(Strix Halo),Radeon 8060S(gfx1151, 40CU) | `lscpu \| grep -i "AI Max"` |
| 内存 | 128GB LPDDR5X 统一内存(UMA),实测可用 GTT ~112GB | `free -g` |
| 系统 | Ubuntu 26.04 LTS(内核 ≥ 7.0,含 amdgpu) | `lsb_release -d; uname -r` |
| 驱动 | Mesa ≥ 26.0(RADV),认得 gfx1151 | 见下 |
| 磁盘 | ≥ 150GB 空闲(模型 ~104GB + 编译) | `df -h /` |

**驱动自检(必须看到 RADV STRIX_HALO,否则会掉软渲染 llvmpipe,速度崩到个位数):**
```bash
sudo apt install -y vulkan-tools
vulkaninfo --summary | grep -iE "deviceName|driverName"
# 期望:
#   deviceName = Radeon 8060S Graphics (RADV STRIX_HALO)
#   driverName = radv
```
若 Mesa 过旧不认 gfx1151:`sudo add-apt-repository ppa:kisak/kisak-mesa && sudo apt update && sudo apt upgrade -y && sudo reboot`

---

## 二、内核参数(GRUB)——两条都要,漏了会出事

编辑 `/etc/default/grub`,把 `GRUB_CMDLINE_LINUX_DEFAULT` 设为:
```
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.lockup_timeout=60000 amdgpu.gttsize=114688 ttm.pages_limit=29360128 ttm.page_pool_size=29360128"
```
然后 `sudo update-grub && sudo reboot`。

- `amdgpu.gttsize=114688 ttm.pages_limit=... ttm.page_pool_size=...`:**打开 112GB 大 GTT 显存**。不加的话核显只能用一小块内存,装不下 125B 模型。
- `amdgpu.lockup_timeout=60000`:**把 GPU"无响应判死"阈值从默认 10 秒放宽到 60 秒**。血泪教训:长上下文(数万 token)的大批次计算,单批就可能超 10 秒,内核会误判 GPU 挂死并强制重置,导致 `ErrorDeviceLost` 满屏。这条是必须的。

验证(重启后):`cat /proc/cmdline | grep -o 'amdgpu.lockup_timeout=[0-9]*'`

---

## 三、编译 fork 引擎(最易失败的一关,先啃)

### 3.1 装编译依赖
```bash
sudo apt install -y build-essential cmake git \
  libvulkan-dev glslc glslang-tools spirv-headers spirv-tools \
  libcurl4-openssl-dev
```
(`spirv-headers` 容易漏——cmake 配置到 Vulkan shader 阶段会报 "Add SPIRV-Headers to CMAKE_PREFIX_PATH")

### 3.2 拿源码(服务器多半直连不了 GitHub)
Strix Halo 机器一般在国内网络,服务器直连 github.com 会失败。可靠办法:**在一台能连 GitHub 的机器上 `git clone`(git 协议比 archive tarball 稳),打包传过去**:
```bash
# 在能上网的机器:
git clone --depth 1 --single-branch --branch "vulkan/qwen4exp-rocmfpx" \
  https://github.com/LaurentZuijdwijk/llama.cpp cafe-fork
tar --exclude='.git' -czf fork-src.tar.gz -C cafe-fork .
# scp/传到目标机 ~/llama/,校验 md5 一致后解压:
mkdir -p ~/llama/cafe-fork && tar -xzf fork-src.tar.gz -C ~/llama/cafe-fork
```
(镜像 `ghfast.top` 前缀有时可用但不稳;git clone 最可靠)

### 3.3 编译
```bash
cd ~/llama/cafe-fork
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build -j$(nproc) --target llama-server   # 32 核约 15 分钟
```
建议用 `setsid nohup ... > /tmp/build.log 2>&1 &` 后台跑,防 SSH 断连中断。

### 3.4 验证引擎
```bash
~/llama/cafe-fork/build/bin/llama-server --list-devices | grep -i vulkan
# 期望:Vulkan0: Radeon 8060S Graphics (RADV STRIX_HALO) (~115000 MiB free)
```

---

## 四、下载模型(HF 镜像,三个文件)

```bash
pip install -U "huggingface_hub[cli]"   # 需要 hf CLI
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
mkdir -p /data/models/flashnext-strix
D=/data/models/flashnext-strix

# 1) 主模型:STRIX_LEAN 专用量化,3 分片 ~99GB
hf download kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-GGUF \
  --include '*STRIX_LEAN*.gguf' --local-dir $D

# 2) 视觉头 mmproj —— 单独下!通配符 *STRIX_LEAN* 会漏掉它(血泪教训)
hf download kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-GGUF \
  --include '*mmproj*' --local-dir $D

# 3) MTP 草稿头 ~3.85GB
hf download agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF \
  --include '*.gguf' --local-dir $D
```

**两个下载坑:**
- **必须 `HF_HUB_DISABLE_XET=1`**。hf 新版默认走 Xet 后端,会绕过镜像直连 xethub 报 401。
- **hf CLI 一次只给一个 `--include` 模式**。同时传多个位置参数(如 `'A' 'B'`)会让 `--include` 被忽略、只下最后一个文件。分开下。
- 下载支持断点续传,中断重跑同一条即可。100GB 走镜像约 2-4 小时。

下完确认:`ls -lh $D/` 应看到 3 个主模型分片(42+42+16GB)+ 1 个 mmproj(~866MB)+ 1 个 MTP 头(~3.9GB)。

---

## 五、systemd 生产化服务

先建 API key 文件(一行一个,`#` 注释):
```bash
echo "$(openssl rand -hex 24)" | sudo tee /etc/llama-api-keys.txt
sudo chmod 640 /etc/llama-api-keys.txt && sudo chown root:youruser /etc/llama-api-keys.txt
```

写 `/etc/systemd/system/llama-flashnext.service`(把 `youruser`/路径换成你的):
```ini
[Unit]
Description=llama.cpp Flash-Next 125B (fork engine, MTP+vision)
After=network-online.target
Wants=network-online.target

[Service]
User=youruser
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$HOME/llama/cafe-fork/build/bin/llama-server \
  -m /data/models/flashnext-strix/Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-00001-of-00003.gguf \
  -md /data/models/flashnext-strix/Qwen3.8-Flash-Next-MTP-Q8_0.gguf \
  --mmproj /data/models/flashnext-strix/mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  -ngl 999 --n-gpu-layers-draft 999 --flash-attn on \
  --ctx-size 196608 --threads 32 --jinja \
  --alias flash-next \
  --host 0.0.0.0 --port 8080 \
  --api-key-file /etc/llama-api-keys.txt
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now llama-flashnext
```
(主模型只需指定第 1 个分片,llama.cpp 自动找齐 2、3 分片)

---

## 六、关键参数详解(勿乱改,每个都是验证过的)

| 参数 | 值 | 作用 / 为什么 |
|---|---|---|
| `-md` + `--spec-type draft-mtp` | — | 挂 MTP 草稿头做投机解码,提速约 30%,接受率 0.8+。**这是 flash-next 提速的核心** |
| `--spec-draft-n-max` | `2` | 每次草稿 token 数。Flash-Next 上 2 最优;设 3-4 反而更慢(草稿越长错得越多)。**注意:官方 27B 上是 3 最优,别混淆** |
| `--spec-draft-p-min` | `0.75` | 置信度门控。散文场景草稿接受率会掉,门控挡掉低置信草稿,防止被拒草稿倒亏 |
| `--flash-attn` | `on` | **本组合(fork + ROCmFP4 量化)上开 fa 是对的**。⚠️ 但官方主线 + 通用量化上 fa on 反而慢 50%,是另一回事,别套用 |
| `--threads` | `32` | 收益最大的一刀(16核32线程拉满)。默认 16 会让并发掉一半 |
| `--ctx-size` | `196608` | **单请求上下文上限,192K**。见下节"上下文与并发" |
| `-ngl 999 --n-gpu-layers-draft 999` | — | 主模型和草稿头全部层进 GPU |
| `--jinja` | — | 用 jinja 聊天模板引擎 |
| `--mmproj` | — | 视觉头,支持看图(发票/报表识别) |

---

## 七、上下文与并发的权衡(实测边界,极重要)

**核心事实:这个 fork 里 `ctx-size` 是"总 KV cache 量",单路可用 = ctx-size ÷ parallel(路数)。** 崩溃硬线在总量 **~512K** 附近(超了会把 GPU 驱动卡死在内核态,进程变 D 状态、kill 无效,只能重启整机)。

实测数据表:

| 配置 | 总 KV | 单路可用 | 显存 | 结果 |
|---|---|---|---|---|
| `--ctx-size 196608`(默认4路) | 196608 | 每路 192K(unified 共享) | 75GB | ✅ 稳,推荐默认 |
| `--ctx-size 262144 --parallel 2` | 262144 | 128K/路 | 77GB | ✅ 能加载 |
| `--ctx-size 262144 --parallel 1` | 262144 | 单路 256K | ~78GB | ⚠️ 加载过、显存够,但重启后未长期验证,上生产前自测 |
| `--ctx-size 524288 --parallel 2` | 524288 | 256K/路 | — | ❌ **GPU 驱动卡死,需重启,勿用** |

**建议:**
- **默认用 `--ctx-size 196608`(4 路 × 192K)**——久经考验,扛过重启和长期运行;
- **单人/单 agent 重度用、要更大单请求上下文**:`--ctx-size 262144 --parallel 1`(单路 256K),牺牲并发换上下文,上生产前自己压测确认稳定;
- **256K 是这台机器的实际天花板**。模型标称 1M/512K 是理论值,实测 512K 就崩,别试。

---

## 八、验证(部署完逐项过)

```bash
KEY=$(cat /etc/llama-api-keys.txt | grep -v '^#' | head -1)

# 1) 服务活着 + 模型加载
systemctl is-active llama-flashnext
curl -s http://localhost:8080/v1/models -H "Authorization: Bearer $KEY"

# 2) 并发/上下文实际值
curl -s http://localhost:8080/slots -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("lanes",len(d),"n_ctx",d[0]["n_ctx"])'

# 3) 文本 + MTP 速度(看日志的 draft acceptance 应 0.8+)
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"flash-next","max_tokens":200,"messages":[{"role":"user","content":"写一段150字介绍杭州西湖"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("speed", round(d["timings"]["predicted_per_second"],1),"t/s")'
journalctl -u llama-flashnext --since '-2 min' | grep 'draft acceptance' | tail -1

# 4) 视觉(发一张图,OpenAI vision 格式 base64)——识别正确即通
# 5) 显存占用健康
cat /sys/class/drm/card*/device/mem_info_gtt_used | sort -rn | head -1 | awk '{printf "%.1f GB / 112\n", $1/1073741824}'
```

达标线:速度 ≥ 25 t/s、draft acceptance ≥ 0.7、显存 < 85GB、看图正确。

---

## 九、故障速查(踩过的坑合集)

| 症状 | 原因 | 处理 |
|---|---|---|
| 速度 < 5 t/s | 掉了软渲染 llvmpipe | `vulkaninfo --summary` 确认是 RADV;检查 `-ngl 999` |
| 加载崩 `invalid vector subscript` | 用了官方主线引擎 | 必须用本 fork(第三节) |
| `ErrorDeviceLost` 满屏,之后每个请求都报 | GPU 计算超时被内核重置,模型 Vulkan 上下文废了 | `sudo systemctl restart llama-flashnext`;根治靠 `amdgpu.lockup_timeout=60000`(第二节)。可加看门狗:systemd timer 每 2 分钟 grep 日志有 ErrorDeviceLost 就重启服务 |
| GPU 驱动卡死、进程 D 状态、kill 无效、显存不释放 | ctx-size 设太大(≥512K 总量)超硬件极限 | **只能重启整机**;之后把 ctx-size 降到 ≤ 262144 |
| `image input is not supported` | 没挂 mmproj 视觉头 | service 加 `--mmproj`(注意 mmproj 文件要单独下,第四节) |
| `exceeds the available context size` | 请求超过 ctx-size | 调大 ctx(看第七节边界)或客户端(如 Trae)拆任务/落地文件 |
| 下载报 401 / cas-server.xethub | 忘了 `HF_HUB_DISABLE_XET=1` | 加环境变量重下 |
| 下载只下到 1 个文件 | hf CLI 多 `--include` 被当位置参数 | 一次一个 `--include` 模式 |
| 服务重启循环 | service 参数写错(多层转义 sed 易坏) | 用 heredoc 重写整个 service 文件,别用 sed 改 ExecStart |

---

## 十、可选加固(生产建议)

- **看门狗**:防 GPU 偶发挂死无人值守。systemd timer 每 2 分钟跑 `journalctl -u llama-flashnext --since '-2 min' | grep -q ErrorDeviceLost && sudo systemctl restart llama-flashnext`。
- **API key 管理**:`/etc/llama-api-keys.txt` 一行一个,每个应用/人发独立 key,泄露只吊销一行,重启生效。
- **静态 IP + 防火墙**:netplan 固定 IP;`ufw allow from <内网段> to any port 8080`,只放行局域网。
- **前端**:局域网网页聊天/agent 可挂 Open WebUI 或 DeepSeek Harness(dsh)指向 `http://<IP>:8080/v1`,模型名填 `flash-next`。dsh 需在 provider 配置里声明 `defaultInput: [text, image]` 才认视觉,`contextWindow` 填 196608(和服务端对齐,防客户端放任超限)。
- **联网检索**:本地模型无原生联网,可自建 SearXNG(docker)+ 一个透明代理(拦 tool_calls→搜索→回填),另起端口。国内环境 SearXNG 引擎多数被墙,baidu 引擎较稳。

---

## 附:速查——一句话记住每个数字

- 内核:`lockup_timeout=60000` + `gttsize=114688`
- 引擎:`LaurentZuijdwijk/llama.cpp @ vulkan/qwen4exp-rocmfpx`,`cmake -DGGML_VULKAN=ON`
- 模型:kingjones777 STRIX_LEAN(主) + agentionai MTP Q8_0(草稿) + mmproj(视觉,单独下)
- 关键参数:`--spec-draft-n-max 2 --spec-draft-p-min 0.75 --flash-attn on --threads 32 --ctx-size 196608`
- 上下文天花板:单路 256K(ctx 262144 + parallel 1),总量别超 512K 否则卡死重启
- 实测:29-40 t/s,接受率 0.8-0.94,显存 75GB
