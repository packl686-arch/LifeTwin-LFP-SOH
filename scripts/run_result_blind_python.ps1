[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$Script,

    [Parameter(Mandatory = $true)]
    [string]$WorkingDirectory,

    [Parameter(Mandatory = $true)]
    [string]$StdoutPath,

    [Parameter(Mandatory = $true)]
    [string]$StderrPath,

    [Parameter(Mandatory = $true)]
    [string]$ExitManifestPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-OutputPath {
    param([Parameter(Mandatory = $true)][string]$Value)

    $full = [System.IO.Path]::GetFullPath($Value)
    $parent = [System.IO.Path]::GetDirectoryName($full)
    if ([string]::IsNullOrWhiteSpace($parent)) {
        throw 'Output path has no parent directory'
    }
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    return $full
}

$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$scriptPath = (Resolve-Path -LiteralPath $Script).Path
$workingPath = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$stdoutFull = Resolve-OutputPath -Value $StdoutPath
$stderrFull = Resolve-OutputPath -Value $StderrPath
$manifestFull = Resolve-OutputPath -Value $ExitManifestPath
$outputs = @($stdoutFull, $stderrFull, $manifestFull)
if (($outputs | Sort-Object -Unique).Count -ne 3) {
    throw 'stdout, stderr, and exit manifest paths must be distinct'
}
if ($outputs | Where-Object { Test-Path -LiteralPath $_ }) {
    throw 'Result-blind wrapper output already exists'
}

$startedUtc = [DateTime]::UtcNow
$exitCode = $null
$launchExceptionClass = $null
try {
    $arguments = @($scriptPath) + @($ScriptArguments)
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $workingPath `
        -RedirectStandardOutput $stdoutFull `
        -RedirectStandardError $stderrFull `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    $exitCode = $process.ExitCode
}
catch {
    $launchExceptionClass = $_.Exception.GetType().Name
}
$finishedUtc = [DateTime]::UtcNow

if (-not (Test-Path -LiteralPath $stdoutFull)) {
    [System.IO.File]::WriteAllBytes($stdoutFull, [byte[]]::new(0))
}
if (-not (Test-Path -LiteralPath $stderrFull)) {
    [System.IO.File]::WriteAllBytes($stderrFull, [byte[]]::new(0))
}
$stdoutItem = Get-Item -LiteralPath $stdoutFull
$stderrItem = Get-Item -LiteralPath $stderrFull
$payload = [ordered]@{
    schema_version = '1.0.0'
    wrapper_status = if ($null -eq $launchExceptionClass) { 'completed' } else { 'launch_failed' }
    process_exit_code = $exitCode
    launch_exception_class = $launchExceptionClass
    started_utc = $startedUtc.ToString('o')
    finished_utc = $finishedUtc.ToString('o')
    elapsed_seconds = ($finishedUtc - $startedUtc).TotalSeconds
    script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $scriptPath).Hash.ToLowerInvariant()
    stdout_byte_count = $stdoutItem.Length
    stdout_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stdoutFull).Hash.ToLowerInvariant()
    stderr_byte_count = $stderrItem.Length
    stderr_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stderrFull).Hash.ToLowerInvariant()
}
$temporaryManifest = "$manifestFull.tmp"
$payload | ConvertTo-Json -Depth 4 -Compress | Set-Content `
    -LiteralPath $temporaryManifest `
    -Encoding utf8NoBOM `
    -NoNewline
[System.IO.File]::Move($temporaryManifest, $manifestFull, $false)

if ($null -ne $launchExceptionClass) {
    exit 125
}
exit $exitCode
