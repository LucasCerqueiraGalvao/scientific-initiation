[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$CacheRoot = "D:\Caches\scientific-initiation\huggingface",

    [string]$ContainerName = "opt-guarded-benchmark",

    [int]$GpuMemoryStopMiB = 15800,

    [int]$GpuMemoryWarnMiB = 15200,

    [int]$PollSeconds = 2,

    [switch]$Resume,

    [switch]$AllowCodeHashMigration,

    [int]$MaxNewScenarios = 0,

    [switch]$AnalyzeAfter
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$imageName = "scientific-initiation-benchmarks:torch2.11-cu128"

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $repoRoot $Path)
}

function Get-RelativeRepoPath([string]$Path) {
    $base = (Resolve-Path $repoRoot).Path
    $target = if (Test-Path $Path) { (Resolve-Path $Path).Path } else { [System.IO.Path]::GetFullPath($Path) }
    if (-not $base.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $base = $base + [System.IO.Path]::DirectorySeparatorChar
    }
    $relative = ([Uri]$base).MakeRelativeUri([Uri]$target).ToString()
    return [Uri]::UnescapeDataString($relative).Replace("\", "/")
}

function Get-GpuSnapshot {
    $raw = & nvidia-smi `
        --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.graphics `
        --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        throw "Nao foi possivel consultar a GPU com nvidia-smi."
    }
    $parts = @($raw)[0] -split ",\s*"
    if ($parts.Count -ne 7) { throw "Saida inesperada do nvidia-smi: $raw" }
    return [pscustomobject]@{
        Timestamp = $parts[0]
        TemperatureC = [int]$parts[1]
        UtilizationPercent = [int]$parts[2]
        MemoryUsedMiB = [int]$parts[3]
        MemoryTotalMiB = [int]$parts[4]
        PowerDrawW = [double]$parts[5]
        GraphicsClockMHz = [int]$parts[6]
    }
}

function Get-ContainerMemoryMiB([string]$Name) {
    $raw = & docker stats $Name --no-stream --format "{{.MemUsage}}" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
    $used = (($raw -split "/")[0]).Trim()
    if ($used -match "^([0-9.]+)\s*GiB$") { return [double]$Matches[1] * 1024 }
    if ($used -match "^([0-9.]+)\s*MiB$") { return [double]$Matches[1] }
    if ($used -match "^([0-9.]+)\s*KiB$") { return [double]$Matches[1] / 1024 }
    return $null
}

function Test-ContainerRunning([string]$Name) {
    $id = & docker ps --filter "name=^/$Name$" --format "{{.ID}}" 2>$null
    return ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($id))
}

$configPath = Resolve-RepoPath $Config
$outputPath = Resolve-RepoPath $OutputDir
$guardOutputPath = "$outputPath.guard"
$monitorPath = Join-Path $guardOutputPath "guard_monitor.csv"
$guardPath = Join-Path $guardOutputPath "guard_status.json"
$dockerStdoutLogPath = Join-Path $guardOutputPath "docker.stdout.log"
$dockerStderrLogPath = Join-Path $guardOutputPath "docker.stderr.log"

if (-not (Test-Path $configPath)) {
    throw "Config nao encontrado: $configPath"
}
if (-not (Test-Path $CacheRoot)) {
    throw "CacheRoot nao encontrado: $CacheRoot"
}
New-Item -ItemType Directory -Force -Path $guardOutputPath | Out-Null

$existingContainer = & docker ps -a --filter "name=^/$ContainerName$" --format "{{.ID}}" 2>$null
if (-not [string]::IsNullOrWhiteSpace($existingContainer)) {
    & docker rm -f $ContainerName *> $null
}

$relativeConfig = Get-RelativeRepoPath $configPath
$relativeOutput = Get-RelativeRepoPath $outputPath
$benchmarkArgs = @(
    "python", "-m", "validacao.benchmark_modelos_opt",
    "--config", $relativeConfig,
    "--output-dir", $relativeOutput
)
if ($Resume) { $benchmarkArgs += "--resume" }
if ($AllowCodeHashMigration) { $benchmarkArgs += "--allow-code-hash-migration" }
if ($MaxNewScenarios -gt 0) { $benchmarkArgs += @("--max-new-scenarios", $MaxNewScenarios.ToString()) }

$dockerArgs = @(
    "run", "--name", $ContainerName,
    "--gpus", "all",
    "--volume", "${repoRoot}:/workspace",
    "--volume", "${CacheRoot}:/cache/huggingface",
    "--env", "HF_HOME=/cache/huggingface",
    "--env", "HF_DATASETS_CACHE=/cache/huggingface/datasets",
    "--env", "HF_HUB_OFFLINE=1",
    "--env", "HF_DATASETS_OFFLINE=1",
    "--env", "TRANSFORMERS_OFFLINE=1",
    "--workdir", "/workspace",
    $imageName
) + $benchmarkArgs

Write-Host "[guard] Iniciando container $ContainerName"
$process = Start-Process -FilePath "docker" -ArgumentList $dockerArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $dockerStdoutLogPath -RedirectStandardError $dockerStderrLogPath

"timestamp,gpu_memory_used_mib,gpu_memory_total_mib,gpu_utilization_percent,gpu_temperature_c,gpu_power_w,gpu_clock_mhz,container_memory_mib,event" |
    Set-Content -Path $monitorPath -Encoding UTF8

$stoppedByGuard = $false
$lastEvent = "started"
while (-not $process.HasExited) {
    Start-Sleep -Seconds $PollSeconds
    $gpu = Get-GpuSnapshot
    $containerMemory = Get-ContainerMemoryMiB $ContainerName
    $event = ""
    if ($gpu.MemoryUsedMiB -ge $GpuMemoryWarnMiB) { $event = "warn_gpu_memory" }
    if ($gpu.MemoryUsedMiB -ge $GpuMemoryStopMiB) {
        $event = "stop_gpu_memory"
        $stoppedByGuard = $true
        Write-Warning ("[guard] VRAM {0}/{1} MiB atingiu limite {2} MiB. Parando container." -f $gpu.MemoryUsedMiB, $gpu.MemoryTotalMiB, $GpuMemoryStopMiB)
        & docker stop $ContainerName *> $null
        break
    }
    if ($event) { $lastEvent = $event }
    $line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f `
        $gpu.Timestamp, $gpu.MemoryUsedMiB, $gpu.MemoryTotalMiB, $gpu.UtilizationPercent,
        $gpu.TemperatureC, $gpu.PowerDrawW, $gpu.GraphicsClockMHz,
        ($(if ($null -eq $containerMemory) { "" } else { [math]::Round($containerMemory, 2) })),
        $event
    Add-Content -Path $monitorPath -Value $line -Encoding UTF8
}

$process.WaitForExit()
$exitCode = $process.ExitCode
$runningAfter = Test-ContainerRunning $ContainerName
if ($runningAfter) {
    Write-Warning "[guard] Container ainda em execucao apos saida do processo Docker. Parando para evitar execucao solta."
    & docker stop $ContainerName *> $null
}

$summary = [ordered]@{
    container = $ContainerName
    config = $relativeConfig
    output_dir = $relativeOutput
    guard_output_dir = Get-RelativeRepoPath $guardOutputPath
    stopped_by_guard = $stoppedByGuard
    gpu_memory_stop_mib = $GpuMemoryStopMiB
    gpu_memory_warn_mib = $GpuMemoryWarnMiB
    docker_exit_code = $exitCode
    last_event = $lastEvent
    completed_at = (Get-Date).ToString("o")
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $guardPath -Encoding UTF8

Write-Host "[guard] Docker exit code: $exitCode"
Write-Host "[guard] Monitor: $monitorPath"
Write-Host "[guard] Status: $guardPath"

if ($AnalyzeAfter -and -not $stoppedByGuard) {
    Write-Host "[guard] Rodando analise do diretorio de saida."
    & docker run --rm --gpus all `
        --volume "${repoRoot}:/workspace" `
        --volume "${CacheRoot}:/cache/huggingface" `
        --env HF_HOME=/cache/huggingface `
        --env HF_DATASETS_CACHE=/cache/huggingface/datasets `
        --env HF_HUB_OFFLINE=1 `
        --env HF_DATASETS_OFFLINE=1 `
        --env TRANSFORMERS_OFFLINE=1 `
        --workdir /workspace `
        $imageName `
        python -m validacao.analise_benchmark_modelos_opt --evidence-dir $relativeOutput
}

exit $exitCode
