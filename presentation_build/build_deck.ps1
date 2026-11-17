$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root 'submission'
$spec = Get-Content -LiteralPath (Join-Path $out 'deck_spec.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$render = Join-Path $out 'slides'
New-Item -ItemType Directory -Path $render -Force | Out-Null
$ppt = New-Object -ComObject PowerPoint.Application
$deck = $ppt.Presentations.Add(0)
$deck.PageSetup.SlideWidth = 960
$deck.PageSetup.SlideHeight = 540
$navy = 3423006
$teal = 10262784
$gray = 7829367

function Text($slide, $text, $x, $y, $w, $h, $size=20, $bold=$false, $color=3423006) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $range = $shape.TextFrame.TextRange
    $range.Text = $text
    $range.Font.Name = 'Malgun Gothic'
    $range.Font.NameFarEast = 'Malgun Gothic'
    $range.Font.Size = $size
    $range.Font.Bold = if ($bold) { -1 } else { 0 }
    $range.Font.Color.RGB = $color
    return $shape
}

function Table($slide, $data, $x, $y, $w, $h) {
    $rows = @($data.rows)
    $headers = @($data.headers)
    $shape = $slide.Shapes.AddTable($rows.Count+1, $headers.Count, $x, $y, $w, $h)
    $table = $shape.Table
    for ($r=1; $r -le $rows.Count+1; $r++) {
        for ($c=1; $c -le $headers.Count; $c++) {
            $cell = $table.Cell($r,$c)
            $range = $cell.Shape.TextFrame.TextRange
            $range.Text = if ($r -eq 1) { [string]$headers[$c-1] } else { [string]$rows[$r-2][$c-1] }
            $range.Font.Name = 'Malgun Gothic'
            $range.Font.NameFarEast = 'Malgun Gothic'
            $range.Font.Size = if ($headers.Count -eq 2) { 16 } else { 18 }
            $range.Font.Color.RGB = if ($r -eq 1) { 16777215 } else { $navy }
            $range.Font.Bold = if ($r -eq 1) { -1 } else { 0 }
            $cell.Shape.Fill.ForeColor.RGB = if ($r -eq 1) { $navy } else { 16382457 }
        }
    }
}

function Chart($slide, $data) { }

try {
    for ($i=0; $i -lt $spec.Count; $i++) {
        $s = $spec[$i]
        $slide = $deck.Slides.Add($i+1,12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.ForeColor.RGB = 16777215
        if($i -eq 0) {
            Text $slide $s.title 55 135 850 90 44 $true | Out-Null
            Text $slide $s.subtitle 58 240 850 50 24 | Out-Null
            Text $slide $s.body 58 325 820 130 22 | Out-Null
        } else {
            Text $slide $s.title 38 28 890 58 30 $true | Out-Null
            if($s.subtitle) { Text $slide $s.subtitle 40 90 880 32 17 $false $gray | Out-Null }
            if($null -ne $s.chart) {
                Chart $slide $s.chart
                Text $slide $s.body 670 165 250 280 19 | Out-Null
            } elseif($null -ne $s.table) {
                if($i -eq 1) {
                    Table $slide $s.table 45 155 865 135
                    Text $slide $s.body 50 325 850 160 21 | Out-Null
                } elseif($i -eq 8) {
                    Table $slide $s.table 40 145 530 300
                    Text $slide $s.body 600 150 320 330 18 | Out-Null
                } else {
                    Table $slide $s.table 45 180 470 170
                    Text $slide $s.body 555 140 365 350 20 | Out-Null
                }
            } else {
                Text $slide $s.body 50 150 860 340 22 | Out-Null
            }
        }
        Text $slide ([string]($i+1)) 895 505 30 22 12 $false $gray | Out-Null
        $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = [string]$s.notes
    }
    $file = Join-Path $out 'GMSC_7min.pptx'
    $deck.SaveAs($file,24)
    $deck.Close()
    python (Join-Path $PSScriptRoot 'insert_native_charts.py')
    if($LASTEXITCODE -ne 0) { throw 'Chart insertion failed' }
    $deck = $ppt.Presentations.Open($file, 0, 0, 0)
    for($i=1; $i -le $deck.Slides.Count; $i++) {
        $deck.Slides.Item($i).Export((Join-Path $render "slide-$i.png"),'PNG',1280,720)
    }
    Write-Output "Saved $file and $($deck.Slides.Count) slide renders."
} finally {
    $deck.Close()
    $ppt.Quit()
}
