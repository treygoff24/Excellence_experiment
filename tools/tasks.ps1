<#
  Windows task runner mirroring Makefile phases.

  - Lists available tasks
  - Runs task commands via local venv Python
  - With -WhatIf, prints the underlying command and exits 0

  Usage examples:
    powershell -ExecutionPolicy Bypass -File tools\tasks.ps1 -List
    powershell -ExecutionPolicy Bypass -File tools\tasks.ps1 -Task Data
    powershell -ExecutionPolicy Bypass -File tools\tasks.ps1 -Task Eval -WhatIf

  Tasks:
    Data, Build, Eval, Parse, Score, Stats, Costs, Report, Plan, Smoke
#>

param(
  [Parameter(Mandatory=$false)] [switch] $List,
  [Parameter(Mandatory=$false)] [string] $Task,
  [Parameter(Mandatory=$false)] [switch] $WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[tasks] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[tasks] ERROR: $msg" -ForegroundColor Red }

try {
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $repoRoot  = Split-Path -Parent $scriptDir
  $venvPy    = Join-Path $repoRoot ".venv\\Scripts\\python.exe"

  $tasks = @{
    'Data'   = @($venvPy, '-m', 'scripts.prepare_data', '--config', 'config/eval_config.yaml');
    'Build'  = @($venvPy, '-m', 'scripts.build_batches', '--config', 'config/eval_config.yaml');
    'Eval'   = @($venvPy, '-m', 'scripts.run_all');
    'Parse'  = @($venvPy, '-m', 'fireworks.parse_results', '--results_jsonl', 'results/results_combined.jsonl', '--out_csv', 'results/predictions.csv');
    'Score'  = @($venvPy, '-m', 'scoring.score_predictions', '--pred_csv', 'results/predictions.csv', '--prepared_dir', 'data/prepared', '--out_dir', 'results');
    'Stats'  = @($venvPy, '-m', 'scoring.stats', '--per_item_csv', 'results/per_item_scores.csv', '--config', 'config/eval_config.yaml', '--out_path', 'results/significance.json');
    'Report' = @($venvPy, '-m', 'scripts.generate_report', '--config', 'config/eval_config.yaml', '--results_dir', 'results', '--reports_dir', 'reports');
    'Costs'  = @($venvPy, '-m', 'scripts.summarize_costs', '--pred_csv', 'results/predictions.csv', '--config', 'config/eval_config.yaml', '--out_path', 'results/costs.json');
    'Plan'   = @($venvPy, '-m', 'scripts.run_all', '--config', 'config/eval_config.yaml', '--plan_only');
    'Smoke'  = @($venvPy, '-m', 'scripts.smoke_orchestration', '--n', '3', '--prompt_set', 'operational_only', '--dry_run');
  }

  if ($List) {
    Write-Host "Available tasks:" -ForegroundColor Green
    $tasks.Keys | Sort-Object | ForEach-Object { Write-Host "  $_" }
    exit 0
  }

  if (-not $Task) {
    Write-Info "No -Task specified. Use -List to see options."
    exit 0
  }

  if (-not $tasks.ContainsKey($Task)) {
    throw "Unknown task '$Task'. Use -List to see available tasks."
  }

  $cmd = $tasks[$Task]

  # Pretty print the command
  $pretty = $cmd | ForEach-Object { if ($_ -match '\\s') { '"' + $_ + '"' } else { $_ } } | Out-String
  Write-Info ("Command: " + ($cmd -join ' '))

  if ($WhatIf) {
    Write-Info "WhatIf: not executing."
    exit 0
  }

  if (-not (Test-Path $venvPy)) {
    throw "Virtual environment Python not found at '$venvPy'. Run tools\\bootstrap.ps1 first."
  }

  # Execute the command
  & $cmd[0] $cmd[1..($cmd.Count-1)]
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) { exit $exitCode }
  exit 0
} catch {
  Write-Err $_
  exit 1
}
