param(
    [string]$QwenRoot = 'F:\qwen',
    [int]$Port = 18510
)

$ErrorActionPreference = 'Stop'
$model = Join-Path $QwenRoot 'models\Qwen3-4B-Q4_K_M.gguf'
$runtime = Join-Path $QwenRoot 'runtime\llama-b11784'
$server = Join-Path $runtime 'llama-server.exe'
$keyFile = Join-Path $QwenRoot 'local_api_key.txt'
$logDir = Join-Path $QwenRoot 'logs'
$alias = 'qwen3-4b-local'

foreach ($file in @($model, $server)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required Qwen file is missing: $file"
    }
}
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
if (-not (Test-Path -LiteralPath $keyFile -PathType Leaf)) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $secret = [BitConverter]::ToString($bytes).Replace('-', '').ToLowerInvariant()
    [System.IO.File]::WriteAllText($keyFile, $secret + [Environment]::NewLine)
}
$secret = ([System.IO.File]::ReadAllText($keyFile)).Trim()
if (-not $secret) { throw "Local Qwen API key file is empty: $keyFile" }

# These variables make the project's existing OpenAI-compatible clients use Qwen
# in the current PowerShell process. Compose uses LOCAL_QWEN_API_KEY below.
$env:LOCAL_QWEN_API_KEY = $secret
$env:DEEPSEEK_API_KEY = $secret
$env:DEEPSEEK_MODEL = $alias
$env:DEEPSEEK_BASE_URL = "http://127.0.0.1:$Port/v1"
$env:DEEPSEEK_TIMEOUT = '180'
$bypass = @('localhost', '127.0.0.1', 'host.docker.internal')
$existingBypass = @($env:NO_PROXY, $env:no_proxy) -join ','
$env:NO_PROXY = ((($existingBypass -split ',') + $bypass) |
    Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() } | Select-Object -Unique) -join ','
$env:no_proxy = $env:NO_PROXY

$baseUrl = "http://127.0.0.1:$Port"
function Get-LocalModels {
    $raw = & curl.exe --noproxy '*' --silent --max-time 2 -H "Authorization: Bearer $secret" "$baseUrl/v1/models"
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
    try { return ($raw | ConvertFrom-Json) } catch { return $null }
}

$models = Get-LocalModels
if ($models) {
    if (@($models.data.id) -notcontains $alias) {
        throw "Port $Port is serving a different model. Stop that service first."
    }
    Write-Output "Qwen is already available at $baseUrl/v1"
    return
}

$arguments = @(
    '--model', $model, '--alias', $alias,
    '--host', '127.0.0.1', '--port', [string]$Port,
    '--ctx-size', '8192', '--n-gpu-layers', '99', '--parallel', '1',
    '--reasoning', 'off', '--temp', '0', '--api-key-file', $keyFile
)
$process = Start-Process -FilePath $server -ArgumentList $arguments -WorkingDirectory $runtime `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'server.out.log') `
    -RedirectStandardError (Join-Path $logDir 'server.err.log') -PassThru
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        throw "Qwen server exited during startup; inspect $logDir\server.err.log"
    }
    $models = Get-LocalModels
    if ($models -and @($models.data.id) -contains $alias) {
        Write-Output "Qwen is ready at $baseUrl/v1 (PID $($process.Id))"
        return
    }
}
throw "Qwen server did not become ready; inspect $logDir\server.err.log"
