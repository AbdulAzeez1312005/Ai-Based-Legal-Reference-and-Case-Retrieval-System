$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ollamaExe = Join-Path $projectRoot "ollama-runtime\ollama.exe"
$modelStore = Join-Path $projectRoot "ollama-models"
$localHome = Join-Path $projectRoot "ollama-home"

if (-not (Test-Path -LiteralPath $ollamaExe)) {
    throw "Project-local Ollama executable was not found: $ollamaExe"
}
if (-not (Test-Path -LiteralPath (Join-Path $modelStore "manifests\registry.ollama.ai\library\qwen2.5\7b"))) {
    Write-Warning "The qwen2.5:7b manifest is not present in the local model store."
}

# Keep Ollama's home, signing identity, and model store inside Milestone-4.
$env:USERPROFILE = $localHome
$env:HOME = $localHome
$env:OLLAMA_HOME = $localHome
$env:OLLAMA_MODELS = $modelStore
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NO_CLOUD = "1"

Write-Host "Starting project-local Ollama."
Write-Host "Home: $localHome"
Write-Host "Model store: $modelStore"
Write-Host "Cloud features are disabled. Keep this terminal open while using the chatbot."
Set-Location $projectRoot
& $ollamaExe serve
