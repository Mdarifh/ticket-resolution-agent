<#
.SYNOPSIS
    Download portable PostgreSQL 16 for Windows into .local\pgsql (no admin rights needed).

.DESCRIPTION
    Uses the official EnterpriseDB binaries zip. Skips pgAdmin, docs and debug symbols.
    .local\ is git-ignored. Afterwards run .\scripts\start-local.ps1.

.EXAMPLE
    .\scripts\setup-postgres.ps1
#>
[CmdletBinding()]
param(
    [string]$Version = "16.15-2"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Local = Join-Path $Root ".local"
$Target = Join-Path $Local "pgsql"
if (Test-Path (Join-Path $Target "bin\pg_ctl.exe")) {
    Write-Host "PostgreSQL is already in $Target"
    return
}
New-Item -ItemType Directory -Force -Path $Local | Out-Null

$url = "https://get.enterprisedb.com/postgresql/postgresql-$Version-windows-x64-binaries.zip"
$zip = Join-Path $Local "pgsql.zip"
Write-Host "Downloading $url (about 320 MB)"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = "SilentlyContinue"  # the progress bar makes Invoke-WebRequest very slow
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing

Write-Host "Extracting to $Target"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$skip = @("pgsql/pgAdmin", "pgsql/doc/", "pgsql/StackBuilder/", "pgsql/symbols/")
$archive = [IO.Compression.ZipFile]::OpenRead($zip)
try {
    foreach ($entry in $archive.Entries) {
        $name = $entry.FullName
        if ($skip | Where-Object { $name.StartsWith($_) }) { continue }
        $dest = Join-Path $Local $name
        if ($name.EndsWith("/")) { New-Item -ItemType Directory -Force -Path $dest | Out-Null; continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    }
} finally {
    $archive.Dispose()
}
Remove-Item $zip
Write-Host "Done: $(& (Join-Path $Target 'bin\postgres.exe') --version)"
