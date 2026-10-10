$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $projectRoot "frontend"
Set-Location $frontend
if (-not (Test-Path -LiteralPath (Join-Path $frontend "node_modules"))) {
    npm install
}
npm run dev
