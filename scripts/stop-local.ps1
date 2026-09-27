<#
.SYNOPSIS
    Stop what scripts\start-local.ps1 started: Streamlit, FastAPI, Demo Shop and PostgreSQL.

.PARAMETER KeepDatabase
    Leave PostgreSQL running (only stop the app services).

.EXAMPLE
    .\scripts\stop-local.ps1
#>
[CmdletBinding()]
param(
    [switch]$KeepDatabase,
    [string]$PgBin = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Local = Join-Path $Root ".local"
$PidFile = Join-Path $Local "pids.json"

if (Test-Path $PidFile) {
    $saved = Get-Content $PidFile -Raw | ConvertFrom-Json
    foreach ($name in @("ui", "api", "demo")) {
        $processId = $saved.$name
        if (-not $processId) { continue }
        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($proc -and $proc.Path -and $proc.Path -like "$Root*") {
            Stop-Process -Id $processId -Force -Confirm:$false
            Write-Host "Stopped $name (pid $processId)"
        } else {
            Write-Host "$name was not running"
        }
    }
    Remove-Item $PidFile
} else {
    Write-Host "No app services recorded (.local\pids.json missing)"
}

if (-not $KeepDatabase) {
    if (-not $PgBin) { $PgBin = Join-Path $Local "pgsql\bin" }
    $PgCtl = Join-Path $PgBin "pg_ctl.exe"
    if (-not (Test-Path $PgCtl)) {
        $onPath = Get-Command pg_ctl.exe -ErrorAction SilentlyContinue
        if ($onPath) { $PgCtl = $onPath.Source }
    }
    $PgData = Join-Path $Local "pgdata"
    if ((Test-Path $PgCtl) -and (Test-Path $PgData)) {
        $ErrorActionPreference = "Continue"  # pg_ctl writes to stderr; Windows PowerShell would throw
        & $PgCtl status -D $PgData *> $null
        if ($LASTEXITCODE -eq 0) {
            & $PgCtl stop -D $PgData -m fast -w *> $null
            if ($LASTEXITCODE -eq 0) { Write-Host "Stopped PostgreSQL" } else { Write-Warning "pg_ctl stop failed (exit $LASTEXITCODE)" }
        } else {
            Write-Host "PostgreSQL was not running"
        }
    }
}
