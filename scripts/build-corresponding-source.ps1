param(
  [string]$Version = "1.0.4",
  [string]$Tag = "v1.0.4",
  [Parameter(Mandatory=$true)][string]$UpstreamSourceZip,
  [Parameter(Mandatory=$true)][string]$UpstreamCheckpoint,
  [Parameter(Mandatory=$true)][string]$Output
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceZip = (Resolve-Path -LiteralPath $UpstreamSourceZip).Path
$checkpoint = (Resolve-Path -LiteralPath $UpstreamCheckpoint).Path
$outputPath = [IO.Path]::GetFullPath($Output)
$package = Get-Content (Join-Path $projectRoot "package.json") -Raw | ConvertFrom-Json
if($package.version -ne $Version){ throw "package.json version $($package.version) does not match $Version" }

$dirty = (& git -C $projectRoot status --porcelain --untracked-files=all) -join "`n"
if($dirty){ throw "Corresponding source must be built from a clean tagged worktree." }
$head = (& git -C $projectRoot rev-parse HEAD).Trim()
$tagCommit = (& git -C $projectRoot rev-list -n 1 $Tag).Trim()
if(-not $tagCommit -or $head -ne $tagCommit){ throw "$Tag must point to the checked-out commit." }

$expectedSource = "b58b1187fb9373f56db2e463a7629cf9f539c4c97ea361c793620396211122da"
$expectedCheckpoint = "3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8"
$sourceHash = (Get-FileHash -LiteralPath $sourceZip -Algorithm SHA256).Hash.ToLower()
$checkpointHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLower()
if($sourceHash -ne $expectedSource){ throw "Upstream source checksum mismatch." }
if($checkpointHash -ne $expectedCheckpoint){ throw "Upstream checkpoint checksum mismatch." }

$scratch = Join-Path $env:TEMP ("NavbeaRVMSource-" + [guid]::NewGuid().ToString("N"))
$treeZip = Join-Path $scratch "tree.zip"
$stage = Join-Path $scratch "stage"
try {
  New-Item -ItemType Directory -Path $scratch,$stage -Force | Out-Null
  & git -C $projectRoot archive --format=zip --prefix="navbea-rvm-$Version/" --output=$treeZip $Tag
  if($LASTEXITCODE -ne 0){ throw "git archive failed" }
  Expand-Archive -LiteralPath $treeZip -DestinationPath $stage -Force
  $treeRoot = Join-Path $stage "navbea-rvm-$Version"
  $upstream = Join-Path $treeRoot "upstream"
  New-Item -ItemType Directory -Path $upstream -Force | Out-Null
  Copy-Item -LiteralPath $sourceZip -Destination (Join-Path $upstream "RobustVideoMatting-v1.0.0-source.zip")
  Copy-Item -LiteralPath $checkpoint -Destination (Join-Path $upstream "rvm_mobilenetv3.pth")
  $manifest = [ordered]@{
    version = $Version
    tag = $Tag
    commit = $head
    license = "GPL-3.0-only"
    upstream = [ordered]@{
      repository = "https://github.com/PeterL1n/RobustVideoMatting"
      release = "v1.0.0"
      sourceSha256 = $sourceHash
      checkpointSha256 = $checkpointHash
      distributedModelSha256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"
    }
  }
  [IO.File]::WriteAllText((Join-Path $treeRoot "CORRESPONDING_SOURCE_MANIFEST.json"),($manifest | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
  New-Item -ItemType Directory -Path (Split-Path $outputPath) -Force | Out-Null
  if(Test-Path -LiteralPath $outputPath){ Remove-Item -LiteralPath $outputPath -Force }
  Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $outputPath -CompressionLevel Optimal
  $hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLower()
  Write-Output "source=$outputPath commit=$head sha256=$hash"
} finally {
  if(Test-Path -LiteralPath $scratch){ Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue }
}
