[CmdletBinding()]
param(
    [switch]$StartDocker,
    [int]$DockerWaitSeconds = 90
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$settingsPath = Join-Path $env:APPDATA "Docker\settings-store.json"
if (Test-Path -LiteralPath $settingsPath) {
    $backupPath = "$settingsPath.bak-$timestamp"
    Copy-Item -LiteralPath $settingsPath -Destination $backupPath -Force

    $settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
    $settings.EnableDockerAI = $false
    $settings.EnableInference = $false
    $settings.UseInference = $false
    $settings.InferenceCanUseGPUVariant = $false
    $settings | Add-Member -NotePropertyName EnableInferenceGPUVariant -NotePropertyValue $false -Force
    $settings | Add-Member -NotePropertyName EnableInferenceTCP -NotePropertyValue $false -Force
    $settings | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $settingsPath -Encoding UTF8

    Write-Host "[docker-prepare] settings backup: $backupPath"
}

Get-Process -Name "Docker Desktop","com.docker.backend","com.docker.proxy","com.docker.service","docker-ai","docker","wsl" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

wsl --shutdown
Start-Sleep -Seconds 3

$runtimePaths = @(
    (Join-Path $env:LOCALAPPDATA "Docker\run"),
    (Join-Path $env:LOCALAPPDATA "docker-secrets-engine")
)

foreach ($runtimePath in $runtimePaths) {
    $parent = Split-Path -Parent $runtimePath
    $resolvedParent = Resolve-Path -LiteralPath $parent -ErrorAction Stop
    if ($resolvedParent.Path -notlike "$env:LOCALAPPDATA*") {
        throw "Caminho inesperado fora de LOCALAPPDATA: $($resolvedParent.Path)"
    }

    if (Test-Path -LiteralPath $runtimePath) {
        $leaf = Split-Path -Leaf $runtimePath
        $stalePath = Join-Path $parent "$leaf-stale-$timestamp"
        Move-Item -LiteralPath $runtimePath -Destination $stalePath -Force
        Write-Host "[docker-prepare] moved $runtimePath -> $stalePath"
    }

    New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
    Write-Host "[docker-prepare] ready $runtimePath"
}

if ($StartDocker) {
    $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerDesktop)) {
        throw "Docker Desktop nao encontrado em $dockerDesktop"
    }

    Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds($DockerWaitSeconds)
    do {
        Start-Sleep -Seconds 5
        $job = Start-Job -ScriptBlock { docker info --format "{{.ServerVersion}}" }
        if (Wait-Job $job -Timeout 10) {
            $version = Receive-Job $job -ErrorAction SilentlyContinue
            Remove-Job $job -Force
            if ($version) {
                Write-Host "[docker-prepare] Docker ready: $version"
                exit 0
            }
        } else {
            Stop-Job $job -ErrorAction SilentlyContinue
            Remove-Job $job -Force -ErrorAction SilentlyContinue
        }
    } while ((Get-Date) -lt $deadline)

    $backendErrorPath = Join-Path $env:LOCALAPPDATA "Docker\backend.error.json"
    if (Test-Path -LiteralPath $backendErrorPath) {
        Write-Host "[docker-prepare] backend.error.json:"
        Get-Content -LiteralPath $backendErrorPath -Raw
    }
    throw "Docker nao ficou pronto em $DockerWaitSeconds segundos."
}
