[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $paperDir "..\..")).Path
$outputDir = Join-Path $repoRoot "output\pdf"
$mainTex = Join-Path $paperDir "main.tex"

Push-Location $paperDir
try {
    foreach ($pass in 1..2) {
        Write-Host "[LaTeX] passagem $pass/2"
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error $mainTex
        if ($LASTEXITCODE -ne 0) {
            throw "pdfLaTeX falhou na passagem $pass. Consulte main.log."
        }
    }

    Write-Host "[Biber] referencias"
    & biber "main"
    if ($LASTEXITCODE -ne 0) {
        throw "Biber falhou. Consulte main.blg."
    }

    foreach ($pass in 3..4) {
        Write-Host "[LaTeX] passagem $pass/4"
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error $mainTex
        if ($LASTEXITCODE -ne 0) {
            throw "pdfLaTeX falhou na passagem $pass. Consulte main.log."
        }
    }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $paperDir "main.pdf") -Destination (Join-Path $outputDir "paper_ic_transformers.pdf") -Force
    Write-Host "[LaTeX] copia final em $(Join-Path $outputDir 'paper_ic_transformers.pdf')"
}
finally {
    Pop-Location
}
