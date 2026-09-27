<#
.SYNOPSIS
    Start the full stack without Docker: PostgreSQL, Demo Shop, FastAPI and Streamlit.

.DESCRIPTION
    The Windows equivalent of `docker compose up`:
      1. PostgreSQL 16 (portable, in .local\pgsql; data in .local\pgdata) on 127.0.0.1
      2. Creates the databases, runs `alembic upgrade head`, builds the knowledge base index once
      3. Starts Demo Shop, FastAPI and Streamlit in the background, and waits for each health check
    Logs go to logs\, process ids to .local\pids.json. Stop everything with scripts\stop-local.ps1.

    Settings come from .env (POSTGRES_*, API_PORT, STREAMLIT_PORT, DEMO_PORT, ...). Secrets
    stay in .env; nothing is written elsewhere except the database itself.

.PARAMETER PgBin
    Folder with pg_ctl.exe/initdb.exe. Default: .local\pgsql\bin, then PostgreSQL on PATH.

.EXAMPLE
    .\scripts\start-local.ps1
#>
[CmdletBinding()]
param(
    [string]$PgBin = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Local = Join-Path $Root ".local"
$Logs = Join-Path $Root "logs"
$PidFile = Join-Path $Local "pids.json"
New-Item -ItemType Directory -Force -Path $Local, $Logs | Out-Null

function Write-Step([string]$Text) { Write-Host "==> $Text" -ForegroundColor Cyan }

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found. Create it first: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

# --- settings: real environment variables win over .env ---------------------------
$DotEnv = @{}
$EnvPath = Join-Path $Root ".env"
if (Test-Path $EnvPath) {
    foreach ($line in Get-Content $EnvPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $DotEnv[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
        }
    }
} else {
    Write-Warning ".env not found: copy .env.example to .env and set OPENAI_API_KEY for LLM-backed runs."
}
function Get-Setting([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $value) { $value = $DotEnv[$Name] }
    if (-not $value) { $value = $Default }
    return $value
}

$PgUser = Get-Setting "POSTGRES_USER" "qa"
$PgPassword = Get-Setting "POSTGRES_PASSWORD" "qa"
$PgDb = Get-Setting "POSTGRES_DB" "qa_agent"
$PgTestDb = Get-Setting "POSTGRES_TEST_DB" "qa_agent_test"
$PgPort = [int](Get-Setting "POSTGRES_PORT" "5432")
$ApiPort = [int](Get-Setting "API_PORT" "8000")
$UiPort = [int](Get-Setting "STREAMLIT_PORT" "8501")
$DemoPort = [int](Get-Setting "DEMO_PORT" "8765")

function Test-PortInUse([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Format-Arg([string]$Value) {
    # Start-Process joins arguments with spaces and does not quote them (the repo path has spaces).
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

function Invoke-Tool([string]$File, [string[]]$Arguments, [string]$LogName) {
    # Runs a console tool to completion with output in logs\<LogName>.*.log and returns its exit
    # code. Avoids two Windows PowerShell 5.1 pitfalls: stderr output turning into terminating
    # errors under ErrorActionPreference=Stop, and waiting on handles inherited by processes the
    # tool leaves running (pg_ctl start -> postgres).
    $proc = Start-Process -FilePath $File -ArgumentList ($Arguments | ForEach-Object { Format-Arg $_ }) `
        -NoNewWindow -PassThru -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $Logs "$LogName.out.log") -RedirectStandardError (Join-Path $Logs "$LogName.err.log")
    $null = $proc.Handle  # makes ExitCode available after WaitForExit
    $proc.WaitForExit()
    return $proc.ExitCode
}

function Invoke-Query([string]$Sql) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Psql -h 127.0.0.1 -p $PgPort -U $PgUser -d postgres -tAc $Sql 2>&1
        return @{ Code = $LASTEXITCODE; Output = ("$out").Trim() }
    } finally {
        $ErrorActionPreference = $previous
    }
}

# --- 1. PostgreSQL -----------------------------------------------------------------
if (-not $PgBin) {
    $PgBin = Join-Path $Local "pgsql\bin"
    if (-not (Test-Path (Join-Path $PgBin "pg_ctl.exe"))) {
        $onPath = Get-Command pg_ctl.exe -ErrorAction SilentlyContinue
        if ($onPath) { $PgBin = Split-Path $onPath.Source }
        else { throw "PostgreSQL not found. Run .\scripts\setup-postgres.ps1 (portable, no admin needed) or install PostgreSQL 16." }
    }
}
$PgCtl = Join-Path $PgBin "pg_ctl.exe"
$Psql = Join-Path $PgBin "psql.exe"
$PgData = Join-Path $Local "pgdata"

if (-not (Test-Path (Join-Path $PgData "PG_VERSION"))) {
    Write-Step "Initialising the PostgreSQL data directory ($PgData)"
    $pwFile = Join-Path $Local "pwfile.tmp"
    [IO.File]::WriteAllText($pwFile, $PgPassword)
    try {
        $code = Invoke-Tool (Join-Path $PgBin "initdb.exe") @("-D", $PgData, "-U", $PgUser, "--pwfile=$pwFile", "--auth=scram-sha-256", "-E", "UTF8", "--locale=C") "initdb"
        if ($code -ne 0) { throw "initdb failed; see logs\initdb.err.log" }
    } finally {
        Remove-Item $pwFile -ErrorAction SilentlyContinue
    }
}

if ((Invoke-Tool $PgCtl @("status", "-D", $PgData) "pg_ctl") -ne 0) {
    if (Test-PortInUse $PgPort) { throw "Port $PgPort is already in use by another program. Set POSTGRES_PORT in .env." }
    Write-Step "Starting PostgreSQL on 127.0.0.1:$PgPort"
    $code = Invoke-Tool $PgCtl @("start", "-D", $PgData, "-l", (Join-Path $Logs "postgres.log"), "-o", "-p $PgPort -c listen_addresses=127.0.0.1", "-w", "-t", "60") "pg_ctl"
    if ($code -ne 0) { throw "PostgreSQL did not start; see logs\postgres.log" }
} else {
    Write-Step "PostgreSQL is already running"
}

$env:PGPASSWORD = $PgPassword
try {
    foreach ($db in @($PgDb, $PgTestDb)) {
        $exists = Invoke-Query "SELECT 1 FROM pg_database WHERE datname = '$db'"
        if ($exists.Code -ne 0) { throw "Cannot connect to PostgreSQL as ${PgUser}: $($exists.Output)" }
        if ($exists.Output -ne "1") {
            Write-Step "Creating database $db"
            $created = Invoke-Query "CREATE DATABASE `"$db`" OWNER `"$PgUser`""
            if ($created.Code -ne 0) { throw "Could not create database ${db}: $($created.Output)" }
        }
    }
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
}

# --- 2. environment for the app processes (inherited by the children) -------------
$user = [Uri]::EscapeDataString($PgUser)
$password = [Uri]::EscapeDataString($PgPassword)
$env:DATABASE_URL = "postgresql+psycopg://${user}:${password}@127.0.0.1:$PgPort/$PgDb"
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:STREAMLIT_API_BASE_URL = "http://localhost:$ApiPort"
$demoUrl = "http://127.0.0.1:$DemoPort"
$env:API_TEST_BASE_URL = Get-Setting "API_TEST_BASE_URL" $demoUrl
$env:UI_TEST_BASE_URL = Get-Setting "UI_TEST_BASE_URL" $demoUrl

Write-Step "Migrating the database (alembic upgrade head)"
if ((Invoke-Tool $Python @("-m", "alembic", "upgrade", "head") "migrate") -ne 0) {
    throw "Database migration failed; see logs\migrate.err.log"
}

$indexDir = Get-Setting "CHROMA_PERSIST_DIR" "data/chroma"
if (-not (Get-ChildItem -Path $indexDir -Exclude ".gitkeep" -ErrorAction SilentlyContinue)) {
    Write-Step "Building the knowledge base index"
    if ((Invoke-Tool $Python @("-m", "qa_agent.rag.ingestion") "ingestion") -ne 0) {
        Write-Warning "Knowledge base indexing failed (see logs\ingestion.err.log); retrieval stays empty."
    }
}

# --- 3. app services -----------------------------------------------------------------
$pids = @{}
if (Test-Path $PidFile) {
    $saved = Get-Content $PidFile -Raw | ConvertFrom-Json
    foreach ($p in $saved.PSObject.Properties) { $pids[$p.Name] = [int]$p.Value }
}

function Start-AppService([string]$Name, [int]$Port, [string[]]$Arguments, [string]$HealthUrl) {
    $known = $pids[$Name]
    if ($known -and (Get-Process -Id $known -ErrorAction SilentlyContinue) -and (Test-PortInUse $Port)) {
        Write-Step "$Name is already running (pid $known)"
        return
    }
    if (Test-PortInUse $Port) {
        throw "Port $Port ($Name) is already in use by another program. Stop it or change the port in .env."
    }
    Write-Step "Starting $Name on port $Port"
    $proc = Start-Process -FilePath $Python -ArgumentList ($Arguments | ForEach-Object { Format-Arg $_ }) -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Logs "$Name.out.log") -RedirectStandardError (Join-Path $Logs "$Name.err.log") -PassThru
    $pids[$Name] = $proc.Id
    ($pids | ConvertTo-Json) | Set-Content -Path $PidFile -Encoding ascii

    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            Get-Content (Join-Path $Logs "$Name.err.log") -Tail 20 -ErrorAction SilentlyContinue | Write-Host
            throw "$Name exited during startup; see logs\$Name.err.log"
        }
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 700
    }
    throw "$Name did not become healthy at $HealthUrl within 120 s; see logs\$Name.err.log"
}

Start-AppService "demo" $DemoPort @("-m", "qa_agent.tools.demo_server", "--host", "127.0.0.1", "--port", "$DemoPort", "--db", "data/demo_users.db") "$demoUrl/login.html"
Start-AppService "api" $ApiPort @("-m", "uvicorn", "qa_agent.api.main:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "$ApiPort", "--no-server-header") "http://127.0.0.1:$ApiPort/health"
Start-AppService "ui" $UiPort @("-m", "streamlit", "run", "src/qa_agent/ui/Home.py", "--server.address=127.0.0.1", "--server.port=$UiPort", "--server.headless=true", "--browser.gatherUsageStats=false") "http://127.0.0.1:$UiPort/_stcore/health"

# --- summary -------------------------------------------------------------------------
$status = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/system/status" -TimeoutSec 5
Write-Host ""
Write-Host "All services are healthy." -ForegroundColor Green
Write-Host "  Chat app     http://localhost:$ApiPort/app/"
Write-Host "  Dashboard    http://localhost:$UiPort"
Write-Host "  API docs     http://localhost:$ApiPort/docs"
Write-Host "  Demo Shop    http://localhost:$DemoPort   (registered.user@example.com / N3w-Passw0rd!)"
Write-Host "  PostgreSQL   127.0.0.1:$PgPort / $PgDb (user $PgUser)"
Write-Host "  Database connected: $($status.database_configured) | LLM configured: $($status.llm_configured) ($($status.llm_model))"
foreach ($w in $status.warnings) { Write-Warning $w }
Write-Host "  Logs: logs\   Stop: .\scripts\stop-local.ps1"
