Write-Host "Running pre-deploy checks..." -ForegroundColor Cyan
python scripts/preflight_check.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pre-deploy checks failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Pre-deploy checks passed." -ForegroundColor Green
