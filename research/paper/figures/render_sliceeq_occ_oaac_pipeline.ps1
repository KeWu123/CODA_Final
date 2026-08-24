param(
    [string]$EdgePath = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
)

$ErrorActionPreference = 'Stop'
$figureDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath = Join-Path $figureDir 'render_sliceeq_occ_oaac_pipeline.html'
$pngPath = Join-Path $figureDir 'fig_sliceeq_occ_oaac_pipeline.png'
$pdfPath = Join-Path $figureDir 'fig_sliceeq_occ_oaac_pipeline.pdf'

if (-not (Test-Path -LiteralPath $EdgePath -PathType Leaf)) {
    throw "Microsoft Edge was not found at: $EdgePath"
}
if (-not (Test-Path -LiteralPath $htmlPath -PathType Leaf)) {
    throw "Missing HTML render wrapper: $htmlPath"
}

$uri = [System.Uri]::new((Resolve-Path -LiteralPath $htmlPath).Path).AbsoluteUri
$profileRoot = Join-Path $env:TEMP ('codex-sliceeq-figure-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $profileRoot | Out-Null

$pngArgs = @(
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    '--force-device-scale-factor=2',
    '--window-size=1600,920',
    "--user-data-dir=$profileRoot",
    "--screenshot=$pngPath",
    $uri
)
$pngProcess = Start-Process -FilePath $EdgePath -ArgumentList $pngArgs -Wait -PassThru -WindowStyle Hidden
if ($pngProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pngPath)) {
    throw "PNG export failed with exit code $($pngProcess.ExitCode)"
}

$pdfArgs = @(
    '--headless=new',
    '--disable-gpu',
    '--no-pdf-header-footer',
    "--user-data-dir=$profileRoot",
    "--print-to-pdf=$pdfPath",
    $uri
)
$pdfProcess = Start-Process -FilePath $EdgePath -ArgumentList $pdfArgs -Wait -PassThru -WindowStyle Hidden
if ($pdfProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pdfPath)) {
    throw "PDF export failed with exit code $($pdfProcess.ExitCode)"
}

Get-Item -LiteralPath $pngPath, $pdfPath | Select-Object FullName, Length, LastWriteTime
