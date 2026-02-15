param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-JsonRequest {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("GET", "POST", "PATCH", "DELETE")][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [hashtable]$Headers,
        [object]$Body
    )

    if ($null -ne $Body) {
        $payload = $Body | ConvertTo-Json -Depth 8
        return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers -ContentType "application/json" -Body $payload
    }
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers
}

Write-Host "[Smoke] Healthcheck..."
$health = Invoke-JsonRequest -Method GET -Uri "$BaseUrl/api/healthcheck"
if (-not $health.ok) {
    throw "Healthcheck failed"
}

Write-Host "[Smoke] Login..."
$login = Invoke-JsonRequest -Method POST -Uri "$BaseUrl/api/auth/login" -Body @{ username = "doctor"; password = "doctor123" }
if (-not $login.access_token) {
    throw "Auth login did not return access token"
}
$headers = @{ Authorization = "Bearer $($login.access_token)" }

Write-Host "[Smoke] /api/auth/me..."
$me = Invoke-JsonRequest -Method GET -Uri "$BaseUrl/api/auth/me" -Headers $headers
if ($me.username -ne "doctor") {
    throw "Unexpected auth/me user: $($me.username)"
}

Write-Host "[Smoke] /api/predict..."
$patient = @{
    Patient_ID = "SMOKE-$(Get-Random -Minimum 1000 -Maximum 9999)"
    Age = 66
    Gender = "Male"
    Symptoms = "chest pain,confusion"
    "Blood Pressure" = "162/96"
    "Heart Rate" = 126
    Temperature = 101.4
    "Pre-Existing Conditions" = "heart disease,hypertension"
}

$prediction = Invoke-JsonRequest -Method POST -Uri "$BaseUrl/api/predict" -Headers $headers -Body $patient
if (-not $prediction.prediction.risk_level) {
    throw "Prediction response missing risk_level"
}

Write-Host "[Smoke] /api/queue..."
$queue = Invoke-JsonRequest -Method GET -Uri "$BaseUrl/api/queue?status=waiting" -Headers $headers
if ($null -eq $queue.waiting_count) {
    throw "Queue response missing waiting_count"
}

$result = [PSCustomObject]@{
    username = $me.username
    role = $me.role
    risk_level = $prediction.prediction.risk_level
    confidence = $prediction.prediction.confidence
    manual_review_recommended = $prediction.manual_review_recommended
    queue_waiting = $queue.waiting_count
    alert_count = @($queue.alerts).Count
}

Write-Host "[Smoke] Passed"
$result | ConvertTo-Json -Depth 6
