param(
  [string]$Version = "1.0.4",
  [string]$Tag = "v1.0.4",
  [Parameter(Mandatory=$true)][string]$UpstreamSourceZip,
  [Parameter(Mandatory=$true)][string]$UpstreamCheckpoint,
  [Parameter(Mandatory=$true)][string]$SubjectSourceZip,
  [Parameter(Mandatory=$true)][string]$SubjectCheckpoint,
  [Parameter(Mandatory=$true)][string]$Output,
  [switch]$WorkingTree
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceZip = (Resolve-Path -LiteralPath $UpstreamSourceZip).Path
$checkpoint = (Resolve-Path -LiteralPath $UpstreamCheckpoint).Path
$outputPath = [IO.Path]::GetFullPath($Output)
$package = Get-Content (Join-Path $projectRoot "package.json") -Raw | ConvertFrom-Json
if($package.version -ne $Version){ throw "package.json version $($package.version) does not match $Version" }

$dirty = (& git -C $projectRoot status --porcelain --untracked-files=all) -join "`n"
if($dirty -and -not $WorkingTree){ throw "Corresponding source must be built from a clean tagged worktree." }
if($WorkingTree -and $Version -notmatch '-local$'){throw 'Working-tree source must have an explicit -local version'}
$head = (& git -C $projectRoot rev-parse HEAD).Trim()
if(-not $WorkingTree){
  $tagCommit = (& git -C $projectRoot rev-list -n 1 $Tag).Trim()
  if(-not $tagCommit -or $head -ne $tagCommit){ throw "$Tag must point to the checked-out commit." }
}

$expectedSource = "b58b1187fb9373f56db2e463a7629cf9f539c4c97ea361c793620396211122da"
$expectedCheckpoint = "3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8"
$sourceHash = (Get-FileHash -LiteralPath $sourceZip -Algorithm SHA256).Hash.ToLower()
$checkpointHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLower()
if($sourceHash -ne $expectedSource){ throw "Upstream source checksum mismatch." }
if($checkpointHash -ne $expectedCheckpoint){ throw "Upstream checkpoint checksum mismatch." }
$subjects = Get-Content (Join-Path $projectRoot 'models/subjects.json') -Raw | ConvertFrom-Json
if((Get-FileHash -LiteralPath $SubjectSourceZip).Hash.ToLower() -ne $subjects.source.sha256){throw 'Subject source checksum mismatch'}
if((Get-FileHash -LiteralPath $SubjectCheckpoint).Hash.ToLower() -ne $subjects.checkpoint.sha256){throw 'Subject checkpoint checksum mismatch'}

$scratch = Join-Path $env:TEMP ("NavbeaRVMSource-" + [guid]::NewGuid().ToString("N"))
$treeZip = Join-Path $scratch "tree.zip"
$stage = Join-Path $scratch "stage"
try {
  New-Item -ItemType Directory -Path $scratch,$stage -Force | Out-Null
  $treeRoot = Join-Path $stage "navbea-rvm-$Version"
  $fileHashes=[ordered]@{}
  if($WorkingTree){
    $files=@(& git -C $projectRoot ls-files --cached --others --exclude-standard | Sort-Object -Unique)
    if($LASTEXITCODE -ne 0){throw 'Cannot enumerate corresponding source'}
    foreach($relative in $files){
      if($relative.Replace('\','/') -match '(^|/)(test|tests|__tests__)/|\.(test|spec)\.|(^|/)test[_-]'){continue}
      if($relative -eq 'scripts/cpu-preview.cjs'){continue}
      $file=Join-Path $projectRoot $relative
      if(-not(Test-Path -LiteralPath $file -PathType Leaf)){continue}
      $target=Join-Path $treeRoot $relative
      New-Item -ItemType Directory -Path (Split-Path $target) -Force|Out-Null
      Copy-Item -LiteralPath $file -Destination $target
      $fileHashes[$relative]=(Get-FileHash -LiteralPath $file).Hash.ToLower()
    }
  }else{
    & git -C $projectRoot archive --format=zip --prefix="navbea-rvm-$Version/" --output=$treeZip $Tag
    if($LASTEXITCODE -ne 0){ throw "git archive failed" }
    Expand-Archive -LiteralPath $treeZip -DestinationPath $stage -Force
  }
  $upstream = Join-Path $treeRoot "upstream"
  New-Item -ItemType Directory -Path $upstream -Force | Out-Null
  Copy-Item -LiteralPath $sourceZip -Destination (Join-Path $upstream "RobustVideoMatting-v1.0.0-source.zip")
  Copy-Item -LiteralPath $checkpoint -Destination (Join-Path $upstream "rvm_mobilenetv3.pth")
  Copy-Item -LiteralPath $SubjectSourceZip -Destination (Join-Path $upstream "yolov5-v7-source.zip")
  Copy-Item -LiteralPath $SubjectCheckpoint -Destination (Join-Path $upstream "yolov5n-seg.pt")
  $manifest = [ordered]@{
    version = $Version
    tag = if($WorkingTree){$null}else{$Tag}
    kind = if($WorkingTree){'local-working-tree'}else{'tagged-release'}
    files = $fileHashes
    commit = $head
    license = "GPL-3.0-only"
    subjectModel = $subjects
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
  if(Test-Path -LiteralPath $outputPath){throw 'Source output exists; choose a new immutable output'}
  Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $outputPath -CompressionLevel Optimal
  $hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLower()
  Write-Output "source=$outputPath commit=$head sha256=$hash"
} finally {
  $resolved=[IO.Path]::GetFullPath($scratch)
  if([IO.Path]::GetDirectoryName($resolved) -ne [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') -or [IO.Path]::GetFileName($resolved) -notmatch '^NavbeaRVMSource-[a-f0-9]{32}$'){throw 'Unsafe source cleanup path'}
  if(Test-Path -LiteralPath $resolved){ Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue }
}
