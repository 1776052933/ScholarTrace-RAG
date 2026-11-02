# 本机 Qwen3-4B 部署

模型在 `F:\qwen\models\Qwen3-4B-Q4_K_M.gguf`，llama.cpp CUDA 程序在 `F:\qwen\runtime\llama-b11784`。服务只监听本机 `127.0.0.1:18510`；Docker Desktop 容器通过 `host.docker.internal:18510` 访问。访问密钥保存在 `F:\qwen\local_api_key.txt`，不要提交到仓库。

## 启动

在 PowerShell 中进入项目目录，执行：

```powershell
.\scripts\start_local_qwen.ps1
```

脚本会启动服务，并在当前 PowerShell 会话中设置项目现有模型客户端使用的 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_TIMEOUT` 和代理绕过配置。关闭终端不会关闭已经启动的服务；重启电脑后需再运行脚本。

如果 PowerShell 禁止执行脚本，可仅在当前终端使用 `Set-ExecutionPolicy -Scope Process Bypass`，再运行上面的命令。

## 项目调用

直接从本机 Python 运行项目代码时，先在**同一 PowerShell 会话**执行启动脚本。代码继续通过 `ModelSettings` 和已有的 OpenAI 兼容客户端调用模型，无须修改业务代码。

使用完整 Compose 服务时，先准备项目的 `.env`（MySQL 等配置仍按项目部署说明填写），再在**同一 PowerShell 会话**执行：

```powershell
.\scripts\start_local_qwen.ps1
docker compose -f compose.yaml -f deploy/compose.local-qwen.yaml up -d
```

这个覆盖配置将 API 与 Celery worker 的模型入口都指向本机 Qwen，同时把会话提示词上限收紧以适配 8K 上下文窗口。原有环境变量名称中仍保留 `DEEPSEEK_`，这里只表示兼容接口的配置项，实际请求发给本机 Qwen。

## 验证与限制

启动后可在本机执行 `curl.exe --noproxy '*' http://127.0.0.1:18510/health`。需要访问 OpenAI 兼容接口时，使用启动脚本设置的 `DEEPSEEK_API_KEY`。日志在 `F:\qwen\logs`。

如果已有 `scholartrace:local` 镜像，可在启动脚本所在的 PowerShell 会话中验证项目的普通问答和流式问答：

```powershell
docker run --rm -v "$($PWD.Path):/workspace:ro" -w /workspace `
  -e PYTHONPATH=/workspace -e "DEEPSEEK_API_KEY=$env:LOCAL_QWEN_API_KEY" `
  -e DEEPSEEK_MODEL=qwen3-4b-local `
  -e DEEPSEEK_BASE_URL=http://host.docker.internal:18510/v1 `
  -e DEEPSEEK_TIMEOUT=180 `
  -e NO_PROXY=localhost,127.0.0.1,host.docker.internal `
  --entrypoint python scholartrace:local scripts/verify_local_qwen.py
```

本地 4B 量化模型适合开发与小规模测试，回答质量和大模型 API 不等同。当前配置为单并发、8K 上下文，长论文证据和大规模并发仍需单独评估。一次真实的查询理解检查中，模型输出的 JSON 缺少结尾，严格解析失败；项目默认规则回退继续完成了该阶段。因此，当前已验证调用链路和一题 RAG 问答，尚未验证本地模型能稳定完成搜索规划、知识抽取和整套论文工作流。
