# Build the submitted formats from the markdown source:
#   markdown -> docx (pandoc) -> pdf (Word, the format the paper is submitted in)
#
# Word is driven through COM. Windows PowerShell passes arguments to Word's
# optional-by-reference parameters only as plain strings wrapped in [ref], so
# every path is cast before use; a psobject from Join-Path fails at SaveAs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$name = "Weide_FairnessAndAlgorithms_2026"
pandoc "$name.md" -o "$name.docx" --standalone
if ($LASTEXITCODE -ne 0) { throw "pandoc failed" }

$docx = [string](Join-Path $PSScriptRoot "$name.docx")
$pdf = [string](Join-Path $PSScriptRoot "$name.pdf")
if (Test-Path $pdf) { Remove-Item $pdf -Force }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close([ref]0)
} finally {
    $word.Quit([ref]0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

Get-Item "$name.docx", "$name.pdf" | Select-Object Name, Length, LastWriteTime
