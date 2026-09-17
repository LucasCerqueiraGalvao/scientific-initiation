[CmdletBinding()]
param(
    [string]$RunId = "opt-6-7b-complete-guarded-20260916-run2",
    [string]$CacheRoot = "D:\Caches\scientific-initiation\huggingface",
    [int]$GpuMemoryStopMiB = 15800,
    [int]$GpuMemoryWarnMiB = 15200,
    [int]$ExpectedScenarios = 15,
    [int]$MaxAttempts = 20,
    [int]$MaxNewScenariosPerContainer = 1,
    [switch]$SkipAnalysis
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

$config = Join-Path $repoRoot "fases\01_validacao_conceitual\experimentos\modelos_opt_6_7b_incremental_guarded.json"
$outputDir = Join-Path $repoRoot ("resultados\" + $RunId)
$qualityPath = Join-Path $outputDir "quality.csv"
$guardScript = Join-Path $scriptRoot "run_model_benchmark_guarded.ps1"
$imageName = "scientific-initiation-benchmarks:torch2.11-cu128"

function Get-QualityCount {
    if (-not (Test-Path -LiteralPath $qualityPath)) {
        return 0
    }
    return (Import-Csv -LiteralPath $qualityPath | Measure-Object).Count
}

function Get-RelativeDockerPath([string]$Path) {
    return [System.IO.Path]::GetRelativePath($repoRoot, $Path).Replace("\", "/")
}

if (-not (Test-Path -LiteralPath $config)) {
    throw "Config nao encontrado: $config"
}
if (-not (Test-Path -LiteralPath $CacheRoot)) {
    throw "CacheRoot nao encontrado: $CacheRoot"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $before = Get-QualityCount
    Write-Host ("[opt-6.7b] tentativa={0}/{1} cenarios={2}/{3}" -f $attempt, $MaxAttempts, $before, $ExpectedScenarios)

    if ($before -ge $ExpectedScenarios) {
        break
    }

    & $guardScript `
        -Config $config `
        -OutputDir $outputDir `
        -CacheRoot $CacheRoot `
        -ContainerName "opt-6-7b-complete-guarded" `
        -GpuMemoryStopMiB $GpuMemoryStopMiB `
        -GpuMemoryWarnMiB $GpuMemoryWarnMiB `
        -PollSeconds 2 `
        -Resume `
        -AllowCodeHashMigration `
        -MaxNewScenarios $MaxNewScenariosPerContainer

    $exitCode = $LASTEXITCODE
    $after = Get-QualityCount
    Write-Host ("[opt-6.7b] docker_exit={0} cenarios={1}/{2}" -f $exitCode, $after, $ExpectedScenarios)

    if ($after -ge $ExpectedScenarios) {
        break
    }
    if ($after -le $before) {
        throw "A execucao nao registrou novo cenario. Verifique os logs em $outputDir.guard antes de retomar."
    }
    if ($exitCode -ne 0) {
        throw "O container retornou codigo $exitCode apos registrar progresso. Rode novamente com o mesmo comando para retomar."
    }

    Start-Sleep -Seconds 5
}

$completed = Get-QualityCount
if ($completed -lt $ExpectedScenarios) {
    throw "Benchmark incompleto: $completed/$ExpectedScenarios cenarios apos $MaxAttempts tentativas."
}

Write-Host ("[opt-6.7b] benchmark completo: {0}/{1} cenarios." -f $completed, $ExpectedScenarios)

if (-not $SkipAnalysis) {
    $relativeOutput = Get-RelativeDockerPath $outputDir
    Write-Host "[opt-6.7b] rodando analise final."
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
