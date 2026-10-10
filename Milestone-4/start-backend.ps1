$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Virtual environment is missing. Create it from the Milestone-4 folder with: py -3 -m venv .venv"
}
$env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"
Set-Location $projectRoot
& $pythonExe "backend\app.py"
