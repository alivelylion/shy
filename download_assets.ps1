param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv\python.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Existing Python interpreter was not found: $PythonPath"
}
$env:PYTHONNOUSERSITE = '1'
& $PythonPath -s -u (Join-Path $PSScriptRoot 'scripts\download_paper_assets.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonPath -s (Join-Path $PSScriptRoot 'scripts\prepare_public_inputs.py')
exit $LASTEXITCODE
