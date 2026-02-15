# TriAegis - Development Runner
# Starts FastAPI backend (port 8000) and Next.js frontend (port 3000)

Write-Host ''
Write-Host '===========================================' -ForegroundColor Cyan
Write-Host '   TriAegis - Development Environment' -ForegroundColor Cyan
Write-Host '===========================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  Backend  -> http://localhost:8000/docs' -ForegroundColor Green
Write-Host '  Frontend -> http://localhost:3000' -ForegroundColor Green
Write-Host ''
Write-Host '  Press Ctrl+C to stop both servers' -ForegroundColor Yellow
Write-Host ''

# Ensure stale dev servers do not keep old routes/code loaded
$portsToClear = @(8000, 3000)
foreach ($port in $portsToClear) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

$backendExe = Join-Path $PSScriptRoot '.venv\Scripts\uvicorn.exe'
$frontendDir = Join-Path $PSScriptRoot 'frontend'

if (-not (Test-Path $backendExe)) {
    throw "Backend executable not found: $backendExe"
}

$backend = Start-Process -PassThru -NoNewWindow `
    -FilePath $backendExe `
    -ArgumentList @('api:app', '--reload', '--port', '8000') `
    -WorkingDirectory $PSScriptRoot

$frontend = Start-Process -PassThru -NoNewWindow `
    -FilePath 'npm.cmd' `
    -ArgumentList @('run', 'dev') `
    -WorkingDirectory $frontendDir

try {
    while ($true) {
        if ($backend.HasExited -or $frontend.HasExited) {
            Write-Host 'One process exited. Shutting down...' -ForegroundColor Yellow
            break
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontend -and -not $frontend.HasExited) {
        Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host 'Both servers stopped.' -ForegroundColor Green
}
