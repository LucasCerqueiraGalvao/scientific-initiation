[CmdletBinding()]
param(
    [ValidateSet(
        "Status", "Build", "Probe", "Prefetch", "Test", "Smoke", "Synthetic",
        "Hardware", "ModelSmoke", "Models", "Remaining", "All"
    )]
    [string]$Action = "All",
    [string]$CacheRoot = "",
    [string]$RunId = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$imageName = "scientific-initiation-benchmarks:torch2.11-cu128"
$hfCache = if ($CacheRoot) { $CacheRoot } else { Join-Path $repoRoot ".cache\huggingface" }
$resultRoot = Join-Path $repoRoot "resultados"
$gpuUtilizationLimit = 10
$gpuMemoryLimitMiB = 2048
$gpuTemperatureLimitC = 65
$gpuIdlePowerLimitW = 35
$gpuIdleGraphicsClockLimitMHz = 300
$gpuIdleSamples = 5
$gpuIdleIntervalSeconds = 2

if (-not $RunId) {
    $RunId = Get-Date -Format "yyyyMMdd-HHmmss"
}
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$") {
    throw "RunId invalido. Use apenas letras, numeros, ponto, hifen e sublinhado."
}

function Write-Stage([string]$Message) {
    Write-Host "[benchmarks] $Message"
}

function Get-DockerDesktopStatus {
    $raw = & docker desktop status 2>$null | Out-String
    if ($LASTEXITCODE -ne 0) { return "unavailable" }
    if ($raw -match "Status\s+([A-Za-z]+)") { return $Matches[1].ToLowerInvariant() }
    return "unknown"
}

function Assert-Docker {
    $status = Get-DockerDesktopStatus
    if ($status -eq "paused") {
        throw "Docker Desktop esta pausado. Retome o engine antes de executar os benchmarks."
    }
    if ($status -ne "running") {
        Write-Stage "Docker Desktop esta '$status'; iniciando o engine."
        & docker desktop start
        if ($LASTEXITCODE -ne 0) { throw "Falha ao iniciar o Docker Desktop." }
    }
    foreach ($attempt in 1..60) {
        & docker info *> $null
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds 2
    }
    throw "Docker Desktop nao esta em execucao ou o backend Linux nao esta disponivel."
}

function Test-BenchmarkImage {
    & docker image inspect $imageName *> $null
    return $LASTEXITCODE -eq 0
}

function Assert-BenchmarkImage {
    if (-not (Test-BenchmarkImage)) {
        throw "Imagem $imageName ausente. Execute -Action Build primeiro."
    }
}

function Get-GpuSnapshot {
    $raw = & nvidia-smi `
        --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.graphics `
        --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        throw "Nao foi possivel consultar a GPU com nvidia-smi."
    }
    $parts = @($raw)[0] -split ",\s*"
    if ($parts.Count -ne 7) { throw "Saida inesperada do nvidia-smi: $raw" }
    return [pscustomobject]@{
        Name = $parts[0]
        TemperatureC = [int]$parts[1]
        UtilizationPercent = [int]$parts[2]
        MemoryUsedMiB = [int]$parts[3]
        MemoryTotalMiB = [int]$parts[4]
        PowerDrawW = [double]$parts[5]
        GraphicsClockMHz = [int]$parts[6]
    }
}

function Get-GpuIdleState {
    $samples = @()
    foreach ($sampleIndex in 1..$gpuIdleSamples) {
        $samples += Get-GpuSnapshot
        if ($sampleIndex -lt $gpuIdleSamples) { Start-Sleep -Seconds $gpuIdleIntervalSeconds }
    }
    $gameRunning = @(Get-Process -Name "deadlock" -ErrorAction SilentlyContinue).Count -gt 0
    $maxUtilization = ($samples | Measure-Object -Property UtilizationPercent -Maximum).Maximum
    $meanUtilization = ($samples | Measure-Object -Property UtilizationPercent -Average).Average
    $maxMemory = ($samples | Measure-Object -Property MemoryUsedMiB -Maximum).Maximum
    $maxTemperature = ($samples | Measure-Object -Property TemperatureC -Maximum).Maximum
    $maxPower = ($samples | Measure-Object -Property PowerDrawW -Maximum).Maximum
    $maxGraphicsClock = ($samples | Measure-Object -Property GraphicsClockMHz -Maximum).Maximum
    return [pscustomobject]@{
        Name = $samples[0].Name
        GameRunning = $gameRunning
        MeanUtilizationPercent = [double]$meanUtilization
        MaxUtilizationPercent = [int]$maxUtilization
        MaxMemoryUsedMiB = [int]$maxMemory
        MaxTemperatureC = [int]$maxTemperature
        MaxPowerDrawW = [double]$maxPower
        MaxGraphicsClockMHz = [int]$maxGraphicsClock
        Ready = (
            -not $gameRunning -and
            $meanUtilization -lt $gpuUtilizationLimit -and
            $maxMemory -le $gpuMemoryLimitMiB -and
            $maxTemperature -lt $gpuTemperatureLimitC
        )
    }
}

function Get-GpuConsumers {
    try {
        $samples = (Get-Counter "\GPU Engine(*)\Utilization Percentage" `
            -SampleInterval 1 -MaxSamples 2 -ErrorAction Stop).CounterSamples |
            Where-Object { $_.CookedValue -gt 1 }
        $engines = foreach ($sample in $samples) {
            if ($sample.InstanceName -match "pid_(\d+)_") {
                [pscustomobject]@{
                    ProcessId = [int]$Matches[1]
                    Utilization = [double]$sample.CookedValue
                }
            }
        }
        return @($engines | Group-Object ProcessId | ForEach-Object {
            $processId = [int]$_.Name
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            [pscustomobject]@{
                Process = if ($process) { $process.ProcessName } else { "encerrado" }
                ProcessId = $processId
                Utilization = [math]::Round(
                    ($_.Group | Measure-Object -Property Utilization -Sum).Sum,
                    1
                )
            }
        } | Sort-Object Utilization -Descending | Select-Object -First 5)
    }
    catch {
        return @()
    }
}

function Assert-IdleGpu([string]$Stage) {
    Write-Stage "Verificando GPU ociosa antes de $Stage ($gpuIdleSamples amostras)."
    $state = Get-GpuIdleState
    Write-Host (
        "GPU={0}; jogo={1}; uso_medio={2:N1}%; pico={3}%; vram_max={4} MiB; temperatura_max={5} C; potencia_max={6:N1} W; clock_max={7} MHz" -f
        $state.Name, $state.GameRunning, $state.MeanUtilizationPercent,
        $state.MaxUtilizationPercent, $state.MaxMemoryUsedMiB, $state.MaxTemperatureC,
        $state.MaxPowerDrawW, $state.MaxGraphicsClockMHz
    )
    $ready = $state.Ready
    if (-not $ready) {
        $consumers = Get-GpuConsumers
        $consumerUtilization = [double](
            ($consumers | Measure-Object -Property Utilization -Sum).Sum
        )
        if ($consumers.Count -gt 0) {
            Write-Host "Maiores consumidores de GPU observados:"
            $consumers | Format-Table -AutoSize | Out-Host
        }
        $wddmIdle = (
            -not $state.GameRunning -and
            $state.MaxMemoryUsedMiB -le $gpuMemoryLimitMiB -and
            $state.MaxTemperatureC -lt $gpuTemperatureLimitC -and
            $state.MaxPowerDrawW -le $gpuIdlePowerLimitW -and
            $state.MaxGraphicsClockMHz -le $gpuIdleGraphicsClockLimitMHz -and
            $consumerUtilization -lt $gpuUtilizationLimit
        )
        if ($wddmIdle) {
            $ready = $true
            Write-Stage (
                "GPU liberada pelo criterio WDDM de baixo consumo " +
                "(processos={0:N1}%, potencia<={1} W, clock<={2} MHz)." -f
                $consumerUtilization, $gpuIdlePowerLimitW, $gpuIdleGraphicsClockLimitMHz
            )
        }
    }
    if (-not $ready) {
        throw (
            "GPU ocupada. Requisitos: jogo fechado, uso medio < {0}% ou estado WDDM de baixo consumo, VRAM <= {1} MiB e temperatura < {2} C." -f
            $gpuUtilizationLimit, $gpuMemoryLimitMiB, $gpuTemperatureLimitC
        )
    }
}

function Test-ModelCache {
    $required = @(
        (Join-Path $hfCache "hub\models--facebook--opt-125m\snapshots\27dcfa74d334bc871f3234de431e71c6eeba5dd6"),
        (Join-Path $hfCache "hub\models--facebook--opt-350m\snapshots\08ab08cc4b72ff5593870b5d527cf4230323703c"),
        (Join-Path $hfCache "hub\models--facebook--opt-1.3b\snapshots\3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"),
        (Join-Path $hfCache "datasets\Salesforce___wikitext\wikitext-2-raw-v1\0.0.0\f776294184f13b8ff2337b3841cf9269a6216d1e")
    )
    return @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Container) }).Count -eq 0
}

function Assert-ModelCache {
    if (-not (Test-ModelCache)) {
        throw "Cache OPT/WikiText incompleto em $hfCache. Execute -Action Prefetch com a rede disponivel."
    }
}

function Show-Status {
    $dockerStatus = Get-DockerDesktopStatus
    $imageReady = $false
    if ($dockerStatus -eq "running") { $imageReady = Test-BenchmarkImage }
    $cacheReady = Test-ModelCache
    $gpu = Get-GpuSnapshot
    $gameRunning = @(Get-Process -Name "deadlock" -ErrorAction SilentlyContinue).Count -gt 0
    $drives = Get-PSDrive C, D -ErrorAction SilentlyContinue | ForEach-Object {
        "{0}: {1:N2} GB livres" -f $_.Name, ($_.Free / 1GB)
    }
    Write-Host "Docker Desktop: $dockerStatus"
    Write-Host "Imagem: $(if ($imageReady) { 'pronta' } else { 'ausente ou incompleta' })"
    Write-Host "Cache: $(if ($cacheReady) { 'pronto' } else { 'incompleto' }) ($hfCache)"
    Write-Host (
        "GPU: {0}; uso={1}%; VRAM={2}/{3} MiB; temperatura={4} C; potencia={5:N1} W; clock={6} MHz; jogo={7}" -f
        $gpu.Name, $gpu.UtilizationPercent, $gpu.MemoryUsedMiB,
        $gpu.MemoryTotalMiB, $gpu.TemperatureC, $gpu.PowerDrawW,
        $gpu.GraphicsClockMHz, $gameRunning
    )
    $drives | ForEach-Object { Write-Host $_ }
}

function Invoke-Build {
    Write-Stage "Construindo $imageName."
    & docker build --tag $imageName --file (Join-Path $repoRoot "docker\benchmark\Dockerfile") $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "Falha ao construir a imagem." }
}

function Invoke-Container {
    param([string[]]$Command, [switch]$Offline)
    $arguments = @(
        "run", "--rm", "--gpus", "all",
        "--volume", "${repoRoot}:/workspace",
        "--volume", "${hfCache}:/cache/huggingface",
        "--workdir", "/workspace"
    )
    if ($Offline) {
        $arguments += @(
            "--network", "none",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "HF_DATASETS_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1"
        )
    }
    $arguments += $imageName
    $arguments += $Command
    & docker @arguments
    if ($LASTEXITCODE -ne 0) { throw "Comando no container falhou: $($Command -join ' ')" }
}

function Get-EvidenceDirectory([string]$Prefix) {
    return "resultados/$Prefix-$RunId"
}

function Test-ResumeCheckpoint([string]$RelativeOutput) {
    $hostOutput = Join-Path $repoRoot ($RelativeOutput -replace "/", "\")
    return (
        (Test-Path -LiteralPath $hostOutput -PathType Container) -and
        @(Get-ChildItem -LiteralPath $hostOutput -Force).Count -gt 0
    )
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Test-CompletedSuite([string]$RelativeOutput, [string]$AnalysisDirectory) {
    try {
        $hostOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot ($RelativeOutput -replace "/", "\")))
        if (-not $hostOutput.StartsWith($resultRoot, [StringComparison]::OrdinalIgnoreCase)) { return $false }
        $manifestPath = Join-Path $hostOutput "manifest.json"
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $false }
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if ($manifest.status -ne "complete") { return $false }
        foreach ($name in @($manifest.config, $manifest.environment)) {
            if (-not $name -or -not (Test-Path -LiteralPath (Join-Path $hostOutput $name) -PathType Leaf)) {
                return $false
            }
        }
        if ($null -ne $manifest.runs) {
            foreach ($run in $manifest.runs) {
                if ((Get-Sha256 (Join-Path $hostOutput $run.csv)) -ne $run.csv_sha256) { return $false }
                if ((Get-Sha256 (Join-Path $hostOutput $run.timings)) -ne $run.timings_sha256) { return $false }
            }
        }
        if ($manifest.timings -and (Get-Sha256 (Join-Path $hostOutput $manifest.timings)) -ne $manifest.timings_sha256) {
            return $false
        }
        if ($null -ne $manifest.artifacts) {
            foreach ($artifact in $manifest.artifacts) {
                if ((Get-Sha256 (Join-Path $hostOutput $artifact.path)) -ne $artifact.sha256) { return $false }
            }
        }
        $analysisRoot = [IO.Path]::GetFullPath((Join-Path $hostOutput $AnalysisDirectory))
        if (-not $analysisRoot.StartsWith($hostOutput, [StringComparison]::OrdinalIgnoreCase)) { return $false }
        $analysisManifestPath = Join-Path $analysisRoot "analysis_manifest.json"
        if (-not (Test-Path -LiteralPath $analysisManifestPath -PathType Leaf)) { return $false }
        $analysisManifest = Get-Content -LiteralPath $analysisManifestPath -Raw | ConvertFrom-Json
        if ((Get-Sha256 $manifestPath) -ne $analysisManifest.source_manifest_sha256) { return $false }
        foreach ($artifact in @($analysisManifest.artifacts)) {
            $artifactPath = [IO.Path]::GetFullPath((Join-Path $analysisRoot $artifact.path))
            if (-not $artifactPath.StartsWith($analysisRoot, [StringComparison]::OrdinalIgnoreCase)) { return $false }
            if ((Get-Sha256 $artifactPath) -ne $artifact.sha256) { return $false }
            if ($null -ne $artifact.bytes -and (Get-Item -LiteralPath $artifactPath).Length -ne $artifact.bytes) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Move-StaleAnalysis([string]$RelativeOutput, [string]$AnalysisDirectory) {
    $hostOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot ($RelativeOutput -replace "/", "\")))
    $analysisRoot = [IO.Path]::GetFullPath((Join-Path $hostOutput $AnalysisDirectory))
    if (-not $analysisRoot.StartsWith($hostOutput, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Diretorio de analise fora da saida: $analysisRoot"
    }
    if ((Test-Path -LiteralPath $analysisRoot -PathType Container) -and
        @(Get-ChildItem -LiteralPath $analysisRoot -Force).Count -gt 0) {
        $suffix = Get-Date -Format "yyyyMMdd-HHmmss-fff"
        $archive = Join-Path $hostOutput "$AnalysisDirectory.stale-$suffix"
        if (-not $archive.StartsWith($hostOutput, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Destino de arquivo fora da saida: $archive"
        }
        Move-Item -LiteralPath $analysisRoot -Destination $archive
        Write-Stage "Analise anterior preservada em $archive."
    }
}

function Test-HardwareProbe {
    try {
        $capabilitiesPath = Join-Path $resultRoot "hardware_capabilities.json"
        $operatorsPath = Join-Path $resultRoot "hardware_operators.json"
        if (-not (Test-Path -LiteralPath $operatorsPath -PathType Leaf)) { return $false }
        $capabilities = Get-Content -LiteralPath $capabilitiesPath -Raw | ConvertFrom-Json
        return $capabilities.passed -eq $true -and $capabilities.all_accelerators_passed -eq $true
    }
    catch {
        return $false
    }
}

function Invoke-OperationSuite(
    [string]$Config,
    [string]$Prefix,
    [string]$GateName
) {
    $output = Get-EvidenceDirectory $Prefix
    if ($Resume -and (Test-CompletedSuite $output "analise_v3")) {
        Write-Stage "Etapa ja concluida e integra em $output; nenhuma medicao repetida."
        return
    }
    Assert-IdleGpu $GateName
    $command = @(
        "python", "-m", "validacao.benchmark_operacoes_v3",
        "--config", $Config,
        "--output-dir", $output
    )
    if ($Resume -and (Test-ResumeCheckpoint $output)) {
        Write-Stage "Retomando checkpoint existente em $output."
        $command += "--resume"
    }
    Invoke-Container -Offline -Command $command
    if (Test-CompletedSuite $output "analise_v3") {
        Write-Stage "Analise existente continua integra em $output."
        return
    }
    Move-StaleAnalysis $output "analise_v3"
    Invoke-Container -Offline -Command @(
        "python", "-m", "validacao.analise_benchmark_operacoes_v3",
        "--evidence-dir", $output
    )
    if (-not (Test-CompletedSuite $output "analise_v3")) {
        throw "A etapa terminou, mas os manifestos de $output nao foram validados."
    }
}

function Invoke-ModelSuite(
    [string]$Config,
    [string]$Prefix,
    [string]$GateName
) {
    $output = Get-EvidenceDirectory $Prefix
    if ($Resume -and (Test-CompletedSuite $output "analise_modelos")) {
        Write-Stage "Etapa ja concluida e integra em $output; nenhuma medicao repetida."
        return
    }
    Assert-IdleGpu $GateName
    $command = @(
        "python", "-m", "validacao.benchmark_modelos_opt",
        "--config", $Config,
        "--output-dir", $output
    )
    if ($Resume -and (Test-ResumeCheckpoint $output)) {
        Write-Stage "Retomando checkpoint existente em $output."
        $command += "--resume"
    }
    Invoke-Container -Offline -Command $command
    if (Test-CompletedSuite $output "analise_modelos") {
        Write-Stage "Analise existente continua integra em $output."
        return
    }
    Move-StaleAnalysis $output "analise_modelos"
    Invoke-Container -Offline -Command @(
        "python", "-m", "validacao.analise_benchmark_modelos_opt",
        "--evidence-dir", $output
    )
    if (-not (Test-CompletedSuite $output "analise_modelos")) {
        throw "A etapa terminou, mas os manifestos de $output nao foram validados."
    }
}

if ($Action -eq "Status") {
    Show-Status
    exit 0
}

Assert-Docker
New-Item -ItemType Directory -Force -Path $hfCache, $resultRoot | Out-Null

if ($Action -in @("Build", "All") -or ($Action -eq "Remaining" -and -not (Test-BenchmarkImage))) {
    Invoke-Build
}
elseif ($Action -eq "Remaining") {
    Write-Stage "Imagem Docker ja esta pronta; build ignorado."
}

if ($Action -notin @("Build")) { Assert-BenchmarkImage }

if ($Action -in @("Probe", "Remaining", "All")) {
    if ($Action -eq "Remaining" -and (Test-HardwareProbe)) {
        Write-Stage "Probe de hardware ja foi aprovado; execucao ignorada."
    }
    else {
        Assert-IdleGpu "o probe"
        Invoke-Container -Command @(
            "python", "-m", "validacao.hardware_probe",
            "--output", "resultados/hardware_capabilities.json"
        )
    }
}
if ($Action -in @("Test", "Remaining", "All")) {
    Invoke-Container -Offline -Command @(
        "python", "-m", "pytest", "fases/01_validacao_conceitual/tests", "-q", "-W", "error"
    )
}
if ($Action -in @("Prefetch", "All")) {
    Invoke-Container -Command @(
        "python", "-m", "validacao.benchmark_modelos_opt",
        "--config", "fases/01_validacao_conceitual/experimentos/modelos_opt_v1.json",
        "--prefetch"
    )
}
if ($Action -in @("Smoke", "Remaining", "All")) {
    Invoke-OperationSuite `
        "fases/01_validacao_conceitual/experimentos/smoke_hardware_v3.json" `
        "smoke-hardware-v3" `
        "o smoke fisico"
}
if ($Action -in @("Synthetic", "All")) {
    Invoke-OperationSuite `
        "fases/01_validacao_conceitual/experimentos/robustez_sintetica_v3.json" `
        "robustez-sintetica-v3" `
        "a bateria sintetica"
}
if ($Action -in @("ModelSmoke", "Remaining", "All")) {
    Assert-ModelCache
    Invoke-ModelSuite `
        "fases/01_validacao_conceitual/experimentos/modelos_opt_smoke.json" `
        "modelos-opt-smoke" `
        "o smoke OPT"
}
if ($Action -in @("Hardware", "Remaining", "All")) {
    Invoke-OperationSuite `
        "fases/01_validacao_conceitual/experimentos/hardware_nativo_v3.json" `
        "hardware-nativo-v3" `
        "a bateria fisica"
}
if ($Action -in @("Models", "Remaining", "All")) {
    Assert-ModelCache
    Invoke-ModelSuite `
        "fases/01_validacao_conceitual/experimentos/modelos_opt_v1.json" `
        "modelos-opt-v1" `
        "a bateria OPT"
}
