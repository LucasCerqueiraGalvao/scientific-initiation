[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$mainTex = Join-Path $repoRoot "docs\latex\relatorio_ic.tex"
$buildDir = Join-Path $repoRoot "tmp\pdfs\latex"
$outputDir = Join-Path $repoRoot "output\pdf"
$jobName = "relatorio_ic_transformers"

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Push-Location $repoRoot
try {
    foreach ($pass in 1..3) {
        Write-Host "[LaTeX] passagem $pass/3"
        & xelatex `
            -interaction=nonstopmode `
            -halt-on-error `
            -file-line-error `
            "-jobname=$jobName" `
            "-output-directory=$buildDir" `
            $mainTex
        if ($LASTEXITCODE -ne 0) {
            throw "XeLaTeX falhou na passagem $pass. Consulte $buildDir\$jobName.log"
        }
    }

    $builtPdf = Join-Path $buildDir "$jobName.pdf"
    $finalPdf = Join-Path $outputDir "$jobName.pdf"
    Copy-Item -LiteralPath $builtPdf -Destination $finalPdf -Force
    Write-Host "[LaTeX] PDF gerado em $finalPdf"
}
finally {
    Pop-Location
}
