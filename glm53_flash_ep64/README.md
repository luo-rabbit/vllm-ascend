# GLM5.3-Flash 10T EP64 SWA DSpark7 experiment

这是8机、每机8张 Ascend950 的真实 HTTP 服务压测配置包。
主模型及草稿模型均使用 **dummy 随机参数**，没有真实 `.safetensors` 权重。
它用于观察服务路径和算子性能，不能用于模型精度、生成质量或真实接受率评估。
本次仅准备文件并做静态/命令校验，尚未实际运行8机 EP64。

## 文件

- `models/glm5.3-flash-10T-ep64-l78-w4a4c8-kda8/`：主模型配置、量化描述、完整 tokenizer 等文件。
- `models/draft-glm53-dspark-swa-dummy-spec7/config.json`：当前使用的 SWA 草稿配置。
- `scripts/cluster_config.sh`：8个 IP 占位符及公共参数。
- `scripts/run_node0.sh` 到 `run_node7.sh`：每台机器对应的启动入口。
- `scripts/launch_node.sh`：公共服务命令，支持 `--dry-run`、`--no-profile`。
- `aisbench/config.py`：已填写 EP64 的 tokenizer、服务地址、数据集和结果路径。
- `aisbench/run_aisbench.sh`：使用原压测工具源码，复制必要文件到独立工作目录运行。
- `MANIFEST.json`、`FILES.sha256`：来源、源码基线及交付文件校验信息。

模型文件已解引用原始绝对软链，目录内不依赖 `/mnt/weight` 或原 `configs` 路径。
这不是实际权重包，不应移除 dummy 加载参数。

## 固定配置

| 项目 | 默认值 |
| --- | --- |
| 拓扑 | 8机，每机8卡；TP1 / DP64 / EP64；每机 local DP8 |
| 路由专家 | 448，每个 EP rank 7个；top-k16 |
| 主模型 | 78层，19个 DSA 层、59个 KDA 层；KDA8 |
| 草稿 | DSpark7，Qwen3式 SWA/GQA，3层，窗口2048；不是 MLA 草稿 |
| 权重加载 | 主模型 `--load-format dummy`；草稿 `draft_load_config.load_format=dummy` |
| 量化 | 沿用原完整量化描述；MoE W4A4_MXFP4、KDA8；草稿 BF16 |
| Cache | 主模型 KV FP8、indexer KV FP8；草稿 KV BF16 |
| 调度/图 | MRV2，async scheduling，FULL_DECODE_ONLY，prefix caching |
| MoE开关 | force EPLB=true，shared expert multistream=true，enable_fused_mc2=0 |
| 内存 | MBT512，gpu_memory_utilization0.93，HCCL_BUFFSIZE1024 MiB |
| 长度 | MAX_MODEL_LEN268288；输入262132，输出4096，留有余量 |
| 并发 | 总并发256，即64个 DP 各4请求的目标负载 |
| 请求 | data_num256、request_rate0、repeat_rate1、prefix_test、seed1024 |
| Profiling | torch profiler，record_shapes=true，with_stack=false；手动采集3秒 |
| CANN | 默认使用镜像内置包；不自动安装/加载共享盘的独立 PoolKey 算子 |

`max_num_seqs=64` 是每个 DP engine 的调度上限，不用于人为限制 Running4。
运行时能否稳定4仍取决于 KV 容量、预热、调度和实际资源。
`data_num256` 在256并发时只有一轮正式请求；需要更多轮次可显式增加 `DATA_NUM`。
不要改成 DSpark15：当前基线 KDA recurrent 路径最多支持8个验证 token，对应 DSpark7。

## 环境准备

每台机器在容器内使用适配好的 vLLM / vllm-ascend 和 `.venv-312`。
具体 Git HEAD、原始配置指纹见 `MANIFEST.json`，不是任意上游 main 均可直接运行。
需要匹配的 CANN/ATB、torch_npu、FLA 及我们已有的 DSpark、KDA8适配。
此目录不安装软件、不改变框架源码、不加载旧 DP HCCL 同步补丁。

如果8台机器没有共享文件系统，应分别 clone/copy 此目录到各机器；模型路径由脚本
位置推导，无需使用原机器的绝对模型目录。修改 `cluster_config.sh` 的环境路径以
匹配实际安装。所有节点必须使用相同配置、框架版本及兼容算子。

默认使用镜像自带 PoolKeyIndexer，不保证它包含特定优化。先前独立优化版使用
`ops-transformer 8011a1cd18390e9bad9567cdb6f51d1c38f40b0a` 在 CANN9.2.1 下编译。
该二进制包不包含在本目录，且未自动启用 FLA-first / 独立 PoolKey wrapper。
如果另行替换算子，需记录并验证实际 API、tiling、kernel 的来源。

## 8机启动

在 `scripts/cluster_config.sh` 把“机1ip”到“机8ip”替换为各机器
**data0.3001** 的 IP，不是 SSH 管理 IP。node0对应机1，node7对应机8。
如网卡名不同，相应修改 `NIC_NAME`。

先在机1容器执行 node0，随后尽快启动其余7台。每个终端可前台运行；以下示例使用
`nohup`，实际 PID 和日志由操作者留存，不提供自动清理其他进程的命令。

```bash
cd /mnt/share/l00887471/10t/vllm-ascend/glm53_flash_ep64
export RUN_ID=ep64_swa_share_001  # 八机使用同一个值
mkdir -p runtime/$RUN_ID/launcher_logs
# 机1运行node0；机2运行node1；以此类推
nohup bash scripts/run_node0.sh >runtime/$RUN_ID/launcher_logs/node0.log 2>&1 &
echo $!
```

| 机器 | 脚本 | DP ranks | 模式 |
| --- | --- | --- | --- |
| 机1 | run_node0.sh | 0–7 | API :8000 |
| 机2 | run_node1.sh | 8–15 | headless |
| 机3 | run_node2.sh | 16–23 | headless |
| 机4 | run_node3.sh | 24–31 | headless |
| 机5 | run_node4.sh | 32–39 | headless |
| 机6 | run_node5.sh | 40–47 | headless |
| 机7 | run_node6.sh | 48–55 | headless |
| 机8 | run_node7.sh | 56–63 | headless |

仅查看命令、不加载 CANN 或占用 NPU：

```bash
bash scripts/run_node7.sh --dry-run
```

## AISBench

在**机1容器**执行，服务应已完成初始化且 `/health` 正常：

```bash
cd /mnt/share/l00887471/10t/vllm-ascend/glm53_flash_ep64
export RUN_ID=ep64_swa_share_001
export RUN_ROOT="$PWD/runtime/$RUN_ID/aisbench"
bash aisbench/run_aisbench.sh --dry-run
bash aisbench/run_aisbench.sh
```

实际命令：

```bash
python3 aisbench_test.py --input_len 262132 --output_len 4096 --data_num 256 \
  --concurrency 256 --request_rate 0 --dataset_type prefix_cache --repeat_rate 1 \
  --prefix_test --seed 1024 --dp 64 --npu_num 64
```

依赖原 `aisbench_auto_tools_prefix` 工具目录和已安装的 `benchmark` 源码。
默认 WORK_PATH 精确为 `/mnt/share/l00887471/benchmark`，验证 `ais_bench` 实际源码
与可执行入口后运行；新机器路径不同可在公共配置中修改。
原工具目录的 `config.py` 不会被覆盖。包装脚本会复制必要工具文件到
`runtime/.../helpers_*`，将本目录 `config.py` 放入其中。原 helper 会修改
`WORK_PATH` 下的临时API配置和数据集软链，因此同一 benchmark checkout 不应并行
跑其他压测。wrapper退出码不能代替最终成功/失败请求计数。

`--prefix_test` 先以DP64预热64条前缀、输出1 token，随后才是正式4096输出阶段。
必须等正式阶段、尽量全部 DP Running4，且采集窗口没有新 prefill，再开始 profiling。
显式传入 `--npu_num 64` 用于结果汇总，不影响并发或服务拓扑。

## 手动采集3秒

在机1同一容器内执行；只选一个操作者触发，避免重复 start/stop：

```bash
curl --noproxy '*' --fail-with-body -X POST http://127.0.0.1:8000/start_profile
# 确认start成功后计时3秒；成功start后即使中断也应尝试stop
sleep 3
curl --noproxy '*' --fail-with-body -X POST http://127.0.0.1:8000/stop_profile
```

若修改了 API_PORT，相应修改 URL。stop导出可能较慢，不属于采集3秒的计算。
结果为各机器 `runtime/$RUN_ID/prof_data/nodeN`；无共享盘时需分别收集8机文件。
确认64个rank实际文件非空及时间窗口，不能只根据curl成功认定采集完成。

## 提交

此目录不包含运行日志、profiling、虚拟环境或真实权重；`runtime/` 已忽略。
模型大文件均低于GitHub普通Git单文件100MiB限制，无需为本配置包强制引入LFS。
本次没有执行git add、commit或push。你检查后可按自己的方式提交。

```bash
cd /mnt/share/l00887471/10t/vllm-ascend
git status --short
git add glm53_flash_ep64
git diff --cached --stat
git commit -s -m "docs: add GLM5.3 Flash EP64 DSpark experiment bundle"
git push ljt ljt/glm53-dspark-kda8
```
