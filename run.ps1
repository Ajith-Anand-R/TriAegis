param(
	[switch]$RegenerateData
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
	throw "Virtual environment python not found at $PythonExe"
}

function Invoke-Step {
	param(
		[Parameter(Mandatory = $true)]
		[string]$Name,
		[Parameter(Mandatory = $true)]
		[scriptblock]$Script
	)

	Write-Host "==> $Name"
	& $Script
	if ($LASTEXITCODE -ne 0) {
		throw "Step failed: $Name (exit code: $LASTEXITCODE)"
	}
}

Invoke-Step -Name "Install dependencies" -Script { & $PythonExe -m pip install -r requirements.txt }

if ($RegenerateData -or -not (Test-Path (Join-Path $PSScriptRoot "data\synthetic_patients.csv"))) {
	Invoke-Step -Name "Generate synthetic dataset" -Script { & $PythonExe data/generate_data.py }
} else {
	Write-Host "==> Skipping dataset generation (existing data/synthetic_patients.csv). Use -RegenerateData to force."
}

Invoke-Step -Name "Generate sample documents" -Script { & $PythonExe data/generate_sample_documents.py }
Invoke-Step -Name "Train model" -Script { & $PythonExe models/train_model.py }
Invoke-Step -Name "Run self-check" -Script { & $PythonExe scripts/system_self_check.py }

Write-Host "==> Starting FastAPI backend and Next.js frontend"

$backend = Start-Process -PassThru -NoNewWindow -FilePath "cmd" `
	-ArgumentList "/c cd `"$PSScriptRoot`" && .venv\Scripts\uvicorn.exe api:app --reload --port 8000" `
	-WorkingDirectory $PSScriptRoot

$frontend = Start-Process -PassThru -NoNewWindow -FilePath "cmd" `
	-ArgumentList "/c cd `"$PSScriptRoot\frontend`" && npm run dev" `
	-WorkingDirectory "$PSScriptRoot\frontend"

Write-Host "Backend  -> http://localhost:8000/docs"
Write-Host "Frontend -> http://localhost:3000"

try {
	while ($true) {
		if ($backend.HasExited -or $frontend.HasExited) {
			Write-Host "One process exited. Shutting down..." -ForegroundColor Yellow
			break
		}
		Start-Sleep -Seconds 1
	}
}
finally {
	if (-not $backend.HasExited)  { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
	if (-not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
	Write-Host "Both servers stopped." -ForegroundColor Green
}
