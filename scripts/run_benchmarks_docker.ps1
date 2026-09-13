param(
    [ValidateSet("Build", "Probe", "Prefetch", "Test", "Smoke", "Synthetic", "Hardware", "Models", "All")]
    [string]$Action = "All",
    [string]$CacheRoot = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$imageName = "scientific-initiation-benchmarks:torch2.11-cu128"
$hfCache = if ($CacheRoot) { $CacheRoot } else { Join-Path $repoRoot ".cache\huggingface" }
$resultRoot = Join-Path $repoRoot "resultados"

function Assert-Docker {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop nao esta em execucao ou o backend Linux nao esta disponivel."
    }
}

function Invoke-Build {
    docker build --tag $imageName --file (Join-Path $repoRoot "docker\benchmark\Dockerfile") $repoRoot
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
    if ($Offline) { $arguments += @("--network", "none") }
    $arguments += $imageName
    $arguments += $Command
    & docker @arguments
    if ($LASTEXITCODE -ne 0) { throw "Comando no container falhou: $($Command -join ' ')" }
}

function New-EvidenceName([string]$Prefix) {
    return "resultados/$Prefix-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}

function Invoke-OperationSuite([string]$Config, [string]$Prefix) {
    $output = New-EvidenceName $Prefix
    Invoke-Container -Offline -Command @(
        "python", "-m", "validacao.benchmark_operacoes_v3",
        "--config", $Config,
        "--output-dir", $output
    )
    Invoke-Container -Offline -Command @(
        "python", "-m", "validacao.analise_benchmark_operacoes_v3",
        "--evidence-dir", $output
    )
}

function Invoke-ModelSuite {
    $output = New-EvidenceName "modelos-opt-v1"
    Invoke-Container -Offline -Command @(
        "python", "-m", "validacao.benchmark_modelos_opt",
        "--config", "fases/01_validacao_conceitual/experimentos/modelos_opt_v1.json",
        "--output-dir", $output
    )
    Invoke-Container -Offline -Command @(
        "python", "-m", "validacao.analise_benchmark_modelos_opt",
        "--evidence-dir", $output
    )
}

Assert-Docker
New-Item -ItemType Directory -Force -Path $hfCache, $resultRoot | Out-Null

if ($Action -in @("Build", "All")) { Invoke-Build }
if ($Action -in @("Probe", "All")) {
    Invoke-Container -Command @(
        "python", "-m", "validacao.hardware_probe",
        "--output", "resultados/hardware_capabilities.json"
    )
}
if ($Action -in @("Test", "All")) {
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
if ($Action -in @("Smoke", "All")) {
    Invoke-OperationSuite `
        "fases/01_validacao_conceitual/experimentos/smoke_hardware_v3.json" `
        "smoke-hardware-v3"
}
if ($Action -in @("Synthetic", "All")) {
    Invoke-OperationSuite `
        "fases/01_validacao_conceitual/experimentos/robustez_sintetica_v3.json" `
        "robustez-sintetica-v3"
}
if ($Action -in @("Hardware", "All")) {
    Invoke-OperationSuite `
        "fases/01_validacao_conceitual/experimentos/hardware_nativo_v3.json" `
        "hardware-nativo-v3"
}
if ($Action -in @("Models", "All")) {
    Invoke-ModelSuite
}
