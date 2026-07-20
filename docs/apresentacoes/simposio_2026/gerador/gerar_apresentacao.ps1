param(
    [string]$TemplatePath = (Join-Path (Split-Path -Parent $PSScriptRoot) "template - I simposio de CD.pptx"),
    [string]$ContentPath = (Join-Path $PSScriptRoot "conteudo.json"),
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RgbValue {
    param(
        [int]$Red,
        [int]$Green,
        [int]$Blue
    )

    return $Red + ($Green * 256) + ($Blue * 65536)
}

function Get-ThemeColors {
    param(
        [string]$Theme = "blue"
    )

    switch ($Theme) {
        "green" {
            return @{
                Fill = Get-RgbValue -Red 235 -Green 247 -Blue 240
                Line = Get-RgbValue -Red 34 -Green 125 -Blue 81
                Text = Get-RgbValue -Red 18 -Green 92 -Blue 58
                Detail = Get-RgbValue -Red 74 -Green 103 -Blue 86
                Accent = Get-RgbValue -Red 83 -Green 164 -Blue 117
            }
        }
        "amber" {
            return @{
                Fill = Get-RgbValue -Red 247 -Green 244 -Blue 238
                Line = Get-RgbValue -Red 176 -Green 116 -Blue 55
                Text = Get-RgbValue -Red 124 -Green 77 -Blue 30
                Detail = Get-RgbValue -Red 112 -Green 98 -Blue 82
                Accent = Get-RgbValue -Red 204 -Green 148 -Blue 77
            }
        }
        default {
            return @{
                Fill = Get-RgbValue -Red 240 -Green 246 -Blue 251
                Line = Get-RgbValue -Red 42 -Green 96 -Blue 153
                Text = Get-RgbValue -Red 24 -Green 64 -Blue 114
                Detail = Get-RgbValue -Red 90 -Green 105 -Blue 120
                Accent = Get-RgbValue -Red 110 -Green 145 -Blue 180
            }
        }
    }
}

function Get-ShapeById {
    param(
        $Slide,
        [int]$ShapeId
    )

    foreach ($shape in $Slide.Shapes) {
        if ($shape.Id -eq $ShapeId) {
            return $shape
        }
    }

    throw "Shape id $ShapeId not found on slide $($Slide.SlideIndex)."
}

function Set-ShapeText {
    param(
        $Slide,
        [int]$ShapeId,
        [string]$Text,
        [object]$FontSize = $null,
        [switch]$Bold,
        [object]$Alignment = $null
    )

    $shape = Get-ShapeById -Slide $Slide -ShapeId $ShapeId
    $shape.TextFrame.TextRange.Text = $Text

    if ($null -ne $FontSize) {
        $shape.TextFrame.TextRange.Font.Size = [int]$FontSize
    }

    if ($Bold.IsPresent) {
        $shape.TextFrame.TextRange.Font.Bold = -1
    }

    if ($null -ne $Alignment) {
        $shape.TextFrame.TextRange.ParagraphFormat.Alignment = [int]$Alignment
    }

    return $shape
}

function Set-ParagraphFontSize {
    param(
        $Shape,
        [int]$ParagraphIndex,
        [int]$FontSize,
        [bool]$Bold = $false
    )

    $paragraph = $Shape.TextFrame.TextRange.Paragraphs($ParagraphIndex)
    $paragraph.Font.Size = $FontSize
    $paragraph.Font.Bold = $(if ($Bold) { -1 } else { 0 })
}

function Remove-ShapeIfPresent {
    param(
        $Slide,
        [int]$ShapeId
    )

    foreach ($shape in $Slide.Shapes) {
        if ($shape.Id -eq $ShapeId) {
            $shape.Delete()
            return
        }
    }
}

function Add-Card {
    param(
        $Slide,
        [single]$Left,
        [single]$Top,
        [single]$Width,
        [single]$Height,
        [string]$Title,
        [string]$Detail,
        [hashtable]$ThemeColors
    )

    $card = $Slide.Shapes.AddShape(5, $Left, $Top, $Width, $Height)
    $card.Fill.ForeColor.RGB = $ThemeColors.Fill
    $card.Line.ForeColor.RGB = $ThemeColors.Line
    $card.Line.Weight = 1.5
    $card.TextFrame.MarginLeft = 8
    $card.TextFrame.MarginRight = 8
    $card.TextFrame.MarginTop = 8
    $card.TextFrame.MarginBottom = 8
    $card.TextFrame.TextRange.Text = "$Title`r$Detail"
    $card.TextFrame.TextRange.Font.Name = "Arial"
    $card.TextFrame.TextRange.Font.Size = 13
    $card.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $card.TextFrame.VerticalAnchor = 3
    $card.TextFrame.TextRange.Font.Color.RGB = $ThemeColors.Text
    (Set-ParagraphFontSize -Shape $card -ParagraphIndex 1 -FontSize 16 -Bold $true) | Out-Null
    if ($card.TextFrame.TextRange.Paragraphs().Count -ge 2) {
        (Set-ParagraphFontSize -Shape $card -ParagraphIndex 2 -FontSize 11 -Bold $false) | Out-Null
        $card.TextFrame.TextRange.Paragraphs(2).Font.Color.RGB = $ThemeColors.Detail
    }
    return $card
}

function Add-Arrow {
    param(
        $Slide,
        [single]$Left,
        [single]$Top,
        [single]$Width,
        [single]$Height,
        [int]$FillColor
    )

    $arrow = $Slide.Shapes.AddShape(33, $Left, $Top, $Width, $Height)
    $arrow.Fill.ForeColor.RGB = $FillColor
    $arrow.Line.Visible = 0
    return $arrow
}

function Add-FooterCaption {
    param(
        $Slide,
        [single]$Left,
        [single]$Top,
        [single]$Width,
        [single]$Height,
        [string]$Text
    )

    $box = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $box.TextFrame.TextRange.Text = $Text
    $box.TextFrame.TextRange.Font.Name = "Arial"
    $box.TextFrame.TextRange.Font.Size = 13
    $box.TextFrame.TextRange.Font.Bold = 0
    $box.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $box.TextFrame.TextRange.Font.Color.RGB = (Get-RgbValue -Red 60 -Green 60 -Blue 60)
    return $box
}

function Add-CardsRowVisual {
    param(
        $Slide,
        $Items,
        [string]$Theme = "blue",
        [string]$Footer = ""
    )

    $colors = Get-ThemeColors -Theme $Theme
    $count = [int]$Items.Count
    $width = if ($count -ge 4) { 150 } elseif ($count -eq 3) { 185 } else { 240 }
    $gap = if ($count -ge 4) { 14 } else { 18 }
    $totalWidth = ($count * $width) + (($count - 1) * $gap)
    $left = [single](86 + ((660 - $totalWidth) / 2))
    $top = 366
    $height = 64

    for ($index = 0; $index -lt $count; $index += 1) {
        $cardLeft = $left + ($index * ($width + $gap))
        Add-Card -Slide $Slide -Left $cardLeft -Top $top -Width $width -Height $height `
            -Title $Items[$index].title -Detail $Items[$index].detail -ThemeColors $colors | Out-Null
    }

    if ($Footer) {
        Add-FooterCaption -Slide $Slide -Left 88 -Top 440 -Width 640 -Height 20 -Text $Footer | Out-Null
    }
}

function Add-PipelineVisual {
    param(
        $Slide,
        $Items,
        [string]$Theme = "blue"
    )

    $colors = Get-ThemeColors -Theme $Theme
    $count = [int]$Items.Count
    $width = 148
    $gap = 16
    $totalWidth = ($count * $width) + (($count - 1) * $gap)
    $left = [single](84 + ((660 - $totalWidth) / 2))
    $top = 386
    $height = 58

    for ($index = 0; $index -lt $count; $index += 1) {
        $cardLeft = $left + ($index * ($width + $gap))
        Add-Card -Slide $Slide -Left $cardLeft -Top $top -Width $width -Height $height `
            -Title $Items[$index].title -Detail $Items[$index].detail -ThemeColors $colors | Out-Null

        if ($index -lt ($count - 1)) {
            $arrowLeft = $cardLeft + $width + 2
            Add-Arrow -Slide $Slide -Left $arrowLeft -Top ($top + 18) -Width 12 -Height 24 -FillColor $colors.Accent | Out-Null
        }
    }
}

function Add-ChartImageVisual {
    param(
        $Slide,
        [string]$ImagePath,
        [string]$Caption = ""
    )

    Remove-ShapeIfPresent -Slide $Slide -ShapeId 14
    $picture = $Slide.Shapes.AddPicture($ImagePath, 0, -1, 70, 240, 690, 220)
    $picture.LockAspectRatio = 0
    if ($Caption) {
        Add-FooterCaption -Slide $Slide -Left 74 -Top 462 -Width 680 -Height 18 -Text $Caption | Out-Null
    }
}

function Invoke-ChartBuilder {
    param(
        [string]$ScriptPath
    )

    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        return
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        & py -3.11 $ScriptPath
        return
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        & python $ScriptPath
        return
    }

    throw "No Python launcher found to build the literature chart."
}

function Populate-CoverSlide {
    param(
        $Slide,
        $Meta
    )

    $titleShape = Set-ShapeText -Slide $Slide -ShapeId 6 -Text ($Meta.title + "`r" + $Meta.subtitle)
    Set-ParagraphFontSize -Shape $titleShape -ParagraphIndex 1 -FontSize 30 -Bold $true | Out-Null
    Set-ParagraphFontSize -Shape $titleShape -ParagraphIndex 2 -FontSize 18 -Bold $false | Out-Null
    Set-ShapeText -Slide $Slide -ShapeId 7 -Text ($Meta.author + "`r" + $Meta.advisor) -FontSize 20 | Out-Null
    Set-ShapeText -Slide $Slide -ShapeId 8 -Text $Meta.affiliation -FontSize 18 | Out-Null
    Set-ShapeText -Slide $Slide -ShapeId 9 -Text $Meta.eventLine -FontSize 18 | Out-Null
}

function Populate-OverviewSlide {
    param(
        $Slide,
        $SlideSpec
    )

    Set-ShapeText -Slide $Slide -ShapeId 5 -Text $SlideSpec.title -FontSize 28 -Bold | Out-Null
    Set-ShapeText -Slide $Slide -ShapeId 8 -Text ([string]::Join("`r", $SlideSpec.bullets)) -FontSize 26 | Out-Null
}

function Populate-TextSlide {
    param(
        $Slide,
        $SlideSpec
    )

    Set-ShapeText -Slide $Slide -ShapeId 5 -Text $SlideSpec.title -FontSize 28 -Bold | Out-Null
    $bodyShape = Set-ShapeText -Slide $Slide -ShapeId 8 -Text ([string]::Join("`r", $SlideSpec.bullets))
    $bodyShape.TextFrame.TextRange.Font.Name = "Arial"
    $bodyShape.TextFrame.TextRange.Font.Size = 22
    $bodyShape.TextFrame.TextRange.ParagraphFormat.Alignment = 1
}

function Populate-ClosingSlide {
    param(
        $Slide,
        $SlideSpec
    )

    Set-ShapeText -Slide $Slide -ShapeId 11 -Text $SlideSpec.title -FontSize 34 -Bold | Out-Null
    Set-ShapeText -Slide $Slide -ShapeId 12 -Text $SlideSpec.subtitle -FontSize 22 | Out-Null
}

function Render-Visual {
    param(
        $Slide,
        $Visual,
        [string]$ContentDir
    )

    if ($null -eq $Visual) {
        return
    }

    switch ($Visual.type) {
        "cards_row" {
            Remove-ShapeIfPresent -Slide $Slide -ShapeId 14
            Add-CardsRowVisual -Slide $Slide -Items $Visual.items -Theme $Visual.theme -Footer $Visual.footer
        }
        "pipeline" {
            Remove-ShapeIfPresent -Slide $Slide -ShapeId 14
            Add-PipelineVisual -Slide $Slide -Items $Visual.items -Theme $Visual.theme
        }
        "chart_image" {
            $imagePath = if ([System.IO.Path]::IsPathRooted([string]$Visual.path)) {
                [string]$Visual.path
            } else {
                Join-Path $ContentDir ([string]$Visual.path)
            }
            Add-ChartImageVisual -Slide $Slide -ImagePath $imagePath -Caption $Visual.caption
        }
        default {
            throw "Unsupported visual type: $($Visual.type)"
        }
    }
}

if (-not (Test-Path -LiteralPath $TemplatePath)) {
    throw "Template not found: $TemplatePath"
}

if (-not (Test-Path -LiteralPath $ContentPath)) {
    throw "Content file not found: $ContentPath"
}

$contentDir = Split-Path -Parent $ContentPath
$chartScriptPath = Join-Path $contentDir "gerar_grafico_literatura.py"
Invoke-ChartBuilder -ScriptPath $chartScriptPath

$content = Get-Content -LiteralPath $ContentPath -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not $OutputPath) {
    $OutputPath = Join-Path $PSScriptRoot $content.meta.outputFileName
}

$outputDir = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$workingPath = Join-Path ([System.IO.Path]::GetTempPath()) ("ppt_build_" + [guid]::NewGuid().ToString() + ".pptx")
Copy-Item -LiteralPath $TemplatePath -Destination $workingPath -Force

$powerpoint = $null
$presentation = $null
$generatedSlideCount = 0

try {
    $powerpoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerpoint.Presentations.Open($workingPath, $false, $false, $false)

    $templateSlides = @()
    for ($index = 1; $index -le $presentation.Slides.Count; $index += 1) {
        $templateSlides += $presentation.Slides.Item($index)
    }

    foreach ($slideSpec in $content.slides) {
        $duplicate = $templateSlides[[int]$slideSpec.templateSlide - 1].Duplicate().Item(1)
        $duplicate.MoveTo($presentation.Slides.Count)
    }

    for ($index = $templateSlides.Count; $index -ge 1; $index -= 1) {
        $presentation.Slides.Item($index).Delete()
    }

    for ($index = 1; $index -le $content.slides.Count; $index += 1) {
        $slide = $presentation.Slides.Item($index)
        $slideSpec = $content.slides[$index - 1]

        switch ($slideSpec.kind) {
            "cover" {
                Populate-CoverSlide -Slide $slide -Meta $content.meta
            }
            "overview" {
                Populate-OverviewSlide -Slide $slide -SlideSpec $slideSpec
            }
            "text" {
                Populate-TextSlide -Slide $slide -SlideSpec $slideSpec
            }
            "text_with_visual" {
                Populate-TextSlide -Slide $slide -SlideSpec $slideSpec
                Render-Visual -Slide $slide -Visual $slideSpec.visual -ContentDir $contentDir
            }
            "chart" {
                Populate-TextSlide -Slide $slide -SlideSpec $slideSpec
                Render-Visual -Slide $slide -Visual $slideSpec.visual -ContentDir $contentDir
            }
            "references" {
                Populate-TextSlide -Slide $slide -SlideSpec $slideSpec
            }
            "closing" {
                Populate-ClosingSlide -Slide $slide -SlideSpec $slideSpec
            }
            default {
                throw "Unsupported slide kind: $($slideSpec.kind)"
            }
        }
    }

    $generatedSlideCount = $presentation.Slides.Count
    $presentation.Save()
}
finally {
    if ($presentation) {
        $presentation.Close()
    }
    if ($powerpoint) {
        $powerpoint.Quit()
    }
}

try {
    Copy-Item -LiteralPath $workingPath -Destination $OutputPath -Force
    Write-Output "Presentation generated at: $OutputPath"
    Write-Output "Slide count: $generatedSlideCount"
}
finally {
    if (Test-Path -LiteralPath $workingPath) {
        Remove-Item -LiteralPath $workingPath -Force
    }
}
