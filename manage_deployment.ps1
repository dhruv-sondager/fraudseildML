<#
.SYNOPSIS
    Antigravity Deployment Management Script for FraudShield ML Application
.DESCRIPTION
    Provides start, stop, and status control for the direct Cloudflare Edge deployment.
#>

param(
    [ValidateSet("start", "stop", "status", "restart")]
    [string]$Action = "status"
)

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$statusFile = Join-Path $rootDir "outputs\deployment_info.json"
$supervisorScript = Join-Path $rootDir "deploy_supervisor.py"
$pythonBin = Join-Path $rootDir "venv\Scripts\python.exe"

switch ($Action) {
    "status" {
        Write-Host "================ FraudShield ML Deployment Status ================" -ForegroundColor Cyan
        if (Test-Path $statusFile) {
            $info = Get-Content $statusFile | ConvertFrom-Json
            Write-Host "Status:        $($info.status)" -ForegroundColor Green
            Write-Host "Provider:      $($info.provider)" -ForegroundColor White
            Write-Host "Live URL:      $($info.live_url)" -ForegroundColor Yellow
            Write-Host "Local URL:     $($info.local_url)" -ForegroundColor White
            Write-Host "SSL/TLS:       $($info.ssl_enabled)" -ForegroundColor Green
            Write-Host "Architecture:  $($info.architecture)" -ForegroundColor White
            Write-Host "Deployed At:   $($info.deployed_at)" -ForegroundColor DarkGray
            
            # Check local health
            try {
                $res = Invoke-WebRequest -Uri "$($info.local_url)/_stcore/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
                Write-Host "Health Probe:  ONLINE (HTTP $($res.StatusCode))" -ForegroundColor Green
            } catch {
                Write-Host "Health Probe:  LOCAL SERVICE UNREACHABLE" -ForegroundColor Red
            }
        } else {
            Write-Host "No active deployment found. Run .\manage_deployment.ps1 -Action start" -ForegroundColor Yellow
        }
        Write-Host "==================================================================" -ForegroundColor Cyan
    }

    "stop" {
        Write-Host "Stopping FraudShield ML deployment processes..." -ForegroundColor Yellow
        $killed = 0
        Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue | ForEach-Object {
            Stop-Process -Id $_.Id -Force
            $killed++
        }
        Get-Process | Where-Object { $_.CommandLine -like "*deploy_supervisor.py*" -or $_.CommandLine -like "*main.py*" } -ErrorAction SilentlyContinue | ForEach-Object {
            Stop-Process -Id $_.Id -Force
            $killed++
        }
        if (Test-Path $statusFile) {
            $info = Get-Content $statusFile | ConvertFrom-Json
            $info.status = "STOPPED"
            $info | ConvertTo-Json -Depth 5 | Set-Content $statusFile
        }
        Write-Host "Stopped $killed deployment process(es)." -ForegroundColor Green
    }

    "start" {
        Write-Host "Starting FraudShield ML deployment supervisor..." -ForegroundColor Cyan
        Start-Process -FilePath $pythonBin -ArgumentList $supervisorScript -WorkingDirectory $rootDir
        Write-Host "Deployment supervisor started. Check status in 10-15 seconds with .\manage_deployment.ps1 -Action status" -ForegroundColor Green
    }

    "restart" {
        & $PSCommandPath -Action stop
        Start-Sleep -Seconds 3
        & $PSCommandPath -Action start
    }
}
