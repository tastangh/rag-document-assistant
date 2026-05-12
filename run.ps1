$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Error "Hata: .venv bulunamadi. Once sanal ortam olustur: python -m venv .venv"
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Hata: .venv\Scripts\python.exe bulunamadi."
}

try {
    & $venvPython -c "import streamlit" | Out-Null
} catch {
    Write-Error "Hata: streamlit kurulu degil. Ornek: pip install -r requirements.txt"
}

$env:PYTHONPATH = "$root\src" + ($(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" }))

Write-Host "Belge Asistani baslatiliyor..."
& $venvPython -m streamlit run "src/ui_streamlit.py"

