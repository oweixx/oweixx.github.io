$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
$nodeBinary = if ($nodeCommand) { $nodeCommand.Source } else {
    Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
}
if (-not (Test-Path -LiteralPath $nodeBinary)) {
    Write-Host 'Install Node.js 24 LTS, then run start-preview.cmd again.'
    exit 1
}
$env:PATH = (Split-Path -Parent $nodeBinary) + ';' + $env:PATH

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($npmCommand) {
    $npmBinary = $npmCommand.Source
    $npmPrefix = @()
} else {
    $npmCache = Join-Path $env:LOCALAPPDATA 'oweixx-tools/npm-11'
    $npmEntry = Join-Path $npmCache 'package/bin/npm-cli.js'
    if (-not (Test-Path -LiteralPath $npmEntry)) {
        New-Item -ItemType Directory -Path $npmCache -Force | Out-Null
        $archive = Join-Path $npmCache 'npm.tgz'
        Invoke-WebRequest -Uri 'https://registry.npmjs.org/npm/-/npm-11.6.2.tgz' -OutFile $archive -UseBasicParsing
        & tar.exe -xzf $archive -C $npmCache
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    $npmBinary = $nodeBinary
    $npmPrefix = @($npmEntry)
}

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'node_modules/astro'))) {
    & $npmBinary @npmPrefix ci
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host 'Blog:   http://127.0.0.1:4321/blog/'
Write-Host 'Writer: http://127.0.0.1:4321/write/'
Write-Host 'Keep this window open. Press Ctrl+C to stop.'
& $npmBinary @npmPrefix run dev
exit $LASTEXITCODE
