# Script to run all 8 prompt sets individually for local backend
# This works around the multi-trial limitation in local mode

$prompts = @(
    "operational_only",
    "structure_without_content",
    "philosophy_without_instructions",
    "length_matched_random",
    "length_matched_best_practices",
    "excellence_25_percent",
    "excellence_50_percent",
    "excellence_75_percent"
)

$config = "config\eval_config.local.yaml"
$limitItems = 250
$partsPerDataset = 2

Write-Host "`n🚀 Running all 8 prompt sets individually..." -ForegroundColor Cyan
Write-Host "Each run will process $limitItems items per condition`n" -ForegroundColor Yellow

$totalStart = Get-Date

foreach ($prompt in $prompts) {
    $runStart = Get-Date
    Write-Host "`n[$($prompts.IndexOf($prompt) + 1)/8] Running: $prompt" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Gray
    
    # Clean up working directories before each run to prevent "already done" skips
    Write-Host "Cleaning workspace for fresh run..." -ForegroundColor Yellow
    if (Test-Path "results") {
        Remove-Item -Path "results" -Recurse -Force
    }
    if (Test-Path "data\batch_inputs") {
        Remove-Item -Path "data\batch_inputs" -Recurse -Force
    }
    if (Test-Path "reports") {
        Remove-Item -Path "reports" -Recurse -Force
    }
    
    # Run the evaluation for this prompt set
    python -m scripts.run_all `
        --config $config `
        --limit_items $limitItems `
        --parts_per_dataset $partsPerDataset `
        --prompt_sets $prompt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error running $prompt - exit code: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    
    # Archive the results manually (since --archive doesn't work well with local backend)
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $archiveDir = "experiments\run_$timestamp`_$prompt"
    Write-Host "Archiving results to: $archiveDir" -ForegroundColor Yellow
    
    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
    if (Test-Path "results") {
        Copy-Item -Path "results" -Destination "$archiveDir\results" -Recurse -Force
    }
    if (Test-Path "reports") {
        Copy-Item -Path "reports" -Destination "$archiveDir\reports" -Recurse -Force
    }
    
    $runEnd = Get-Date
    $elapsed = $runEnd - $runStart
    Write-Host "✓ Completed $prompt in $($elapsed.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    Write-Host "  Results saved to: $archiveDir`n" -ForegroundColor Cyan
}

$totalEnd = Get-Date
$totalElapsed = $totalEnd - $totalStart

Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "🎉 All 8 prompt sets completed!" -ForegroundColor Green
Write-Host "Total time: $($totalElapsed.TotalMinutes.ToString('F1')) minutes ($($totalElapsed.TotalHours.ToString('F2')) hours)" -ForegroundColor Cyan
Write-Host "`nResults are in: experiments\run_*\*\results\" -ForegroundColor Yellow

