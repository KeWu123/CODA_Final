param(
    [string]$EdgePath = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
)

$ErrorActionPreference = 'Stop'
$figureDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath = Join-Path $figureDir 'render_sliceeq_occ_oaac_pipeline_v3.html'
$pngPath = Join-Path $figureDir 'fig_sliceeq_occ_oaac_pipeline_v3.png'
$pdfPath = Join-Path $figureDir 'fig_sliceeq_occ_oaac_pipeline_v3.pdf'

if (-not (Test-Path -LiteralPath $EdgePath -PathType Leaf)) { throw "Missing Edge: $EdgePath" }
if (-not (Test-Path -LiteralPath $htmlPath -PathType Leaf)) { throw "Missing HTML: $htmlPath" }

$uri = [System.Uri]::new((Resolve-Path -LiteralPath $htmlPath).Path).AbsoluteUri
$profileRoot = Join-Path $env:TEMP ('codex-sliceeq-v3-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $profileRoot | Out-Null

$pngArgs = @('--headless=new','--disable-gpu','--hide-scrollbars','--force-device-scale-factor=2',
    '--window-size=1600,800',"--user-data-dir=$profileRoot", "--screenshot=$pngPath",$uri)
$p = Start-Process -FilePath $EdgePath -ArgumentList $pngArgs -Wait -PassThru -WindowStyle Hidden
if ($p.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pngPath)) { throw 'PNG export failed' }

$pdfArgs = @('--headless=new','--disable-gpu','--no-pdf-header-footer',
    "--user-data-dir=$profileRoot", "--print-to-pdf=$pdfPath",$uri)
$p = Start-Process -FilePath $EdgePath -ArgumentList $pdfArgs -Wait -PassThru -WindowStyle Hidden
if ($p.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pdfPath)) { throw 'PDF export failed' }

Get-Item -LiteralPath $pngPath,$pdfPath | Select-Object FullName,Length,LastWriteTime
