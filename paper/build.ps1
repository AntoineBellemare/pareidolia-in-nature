# Build paper.pdf locally with tectonic -- no Overleaf, no timeout, free.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File paper\build.ps1
# or just double-click this file, or from the paper/ folder run:  tectonic paper.tex
#
# Requires tectonic on PATH. If "tectonic is not recognized", install it once:
#   winget install TectonicProject.Tectonic      (or: scoop install tectonic)
# then reopen the terminal.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Building paper.pdf with tectonic ..." -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# Full build: XeLaTeX + BibTeX + reruns, all handled by tectonic in one call.
tectonic paper.tex

$sw.Stop()
if ($LASTEXITCODE -eq 0 -and (Test-Path paper.pdf)) {
    Write-Host ("Done in {0:N1}s -> paper.pdf" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    Start-Process paper.pdf   # open the result
} else {
    Write-Host "Build failed -- see the tectonic output above." -ForegroundColor Red
    exit 1
}
