# =============================================================================
# T14 Ablation Runner: beta-C ON vs OFF on CUB-200 (paired, multi-seed)
# =============================================================================
# Per R7 design (docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-design.md)
# and R7 Oracle report Q5 decision: T14 is the BLOCKING gate for beta-C ship.
#
# Protocol:
#   - 3 seeds (42, 123, 456) x 2 conditions (beta-C ON, beta-C OFF) = 6 runs
#   - 1 epoch each on CUB-200
#   - Capture val_acc from training_history.json
#   - Emit t14_results.json with {runs: [{seed, beta_c, val_accuracy}, ...]}
#
# Usage:
#   pwsh -File scripts/run_t14_ablation.ps1
#   pwsh -File scripts/run_t14_ablation.ps1 -Epochs 1 -OutputDir artifacts/t14
#   pwsh -File scripts/run_t14_ablation.ps1 -Seeds @(42,123,456) -Dataset cub200
#
# Expected runtime: 4-6 hours on RTX 4070 (6 runs x ~40-60 min)
# =============================================================================

[CmdletBinding()]
param(
    [int[]]$Seeds = @(42, 123, 456),
    [string]$Dataset = "cub200",
    [int]$Epochs = 1,
    [string]$OutputDir = "artifacts/t14",
    [string]$ExperimentsBase = "experiments/t14",
    [string]$ExtraArgs = "--use-amp",
    [int]$BatchSize = 192,
    [int]$NumWorkers = 4,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------- Paths ----------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Push-Location $ProjectRoot
try {
    Write-Host "=== T14 Ablation Runner ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Dataset:      $Dataset"
    Write-Host "Epochs/run:   $Epochs"
    Write-Host "Seeds:        $($Seeds -join ', ')"
    Write-Host "Output dir:   $OutputDir"
    Write-Host "Experiments:  $ExperimentsBase"
    Write-Host "Extra args:   $ExtraArgs"
    Write-Host ""

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    New-Item -ItemType Directory -Force -Path $ExperimentsBase | Out-Null

    # ---------- Pre-flight: verify uv exists ----------
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv not found on PATH. Install via https://docs.astral.sh/uv/ before running T14."
    }

    # ---------- Pre-flight: verify training script accepts beta-C flag ----------
    # T14 gates beta-C ship. The training script must expose --bias-subtract-lca
    # (and its negation). If the flag is missing, the ablation is meaningless.
    Write-Host "[Pre-flight] Checking for --bias-subtract-lca flag in train_fractal_vit.py..."
    $trainScript = Join-Path $ProjectRoot "src/training/train_fractal_vit.py"
    $trainText = Get-Content $trainScript -Raw -ErrorAction SilentlyContinue
    $hasOn  = $trainText -match "bias[_-]subtract[_-]lca|bias_subtract_lca"
    $hasOff = $trainText -match "no[_-]bias[_-]subtract[_-]lca"
    if (-not $hasOn -or -not $hasOff) {
        Write-Warning "T14 BLOCKED: --bias-subtract-lca / --no-bias-subtract-lca flag not found in $trainScript"
        Write-Warning "Per R7 design (Decision 9c), beta-C MUST be wired into train_fractal_vit.py before T14 runs."
        Write-Warning "Implement beta-C (LCABiasSubtractor) per docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-design.md"
        Write-Warning "Component 2, then re-run this script."
        if (-not $DryRun) { exit 2 }
    } else {
        Write-Host "[Pre-flight] OK: --bias-subtract-lca flag present." -ForegroundColor Green
    }

    # ---------- Run matrix ----------
    # 6 runs: 3 seeds x 2 conditions
    $runs = New-Object System.Collections.Generic.List[object]
    $runIndex = 0
    $totalRuns = $Seeds.Count * 2

    foreach ($seed in $Seeds) {
        foreach ($betaC in @($true, $false)) {
            $runIndex++
            $condition = if ($betaC) { "ON" } else { "OFF" }
            $flag = if ($betaC) { "--bias-subtract-lca" } else { "--no-bias-subtract-lca" }
            $expName = "t14_cub200_s${seed}_betaC${condition}"
            $expDir = Join-Path $ExperimentsBase $expName
            New-Item -ItemType Directory -Force -Path $expDir | Out-Null

            $logFile = Join-Path $OutputDir "${expName}.log"
            $historyFile = Join-Path $expDir "training_history.json"

            Write-Host ""
            Write-Host "[Run $runIndex/$totalRuns] seed=$seed beta-C=$condition" -ForegroundColor Yellow
            Write-Host "  exp:     $expName"
            Write-Host "  log:     $logFile"
            Write-Host "  history: $historyFile"

            $cmd = @(
                "uv", "run", "python", "src/training/train_fractal_vit.py",
                "--dataset", $Dataset,
                "--seed", $seed,
                "--epochs", $Epochs,
                "--batch-size", $BatchSize,
                "--num-workers", $NumWorkers,
                "--experiments-dir", $ExperimentsBase,
                "--experiment-name", $expName,
                $flag
            ) + ($ExtraArgs -split '\s+')

            if ($ExtraArgs.Trim() -eq "") {
                $cmd = $cmd | Where-Object { $_ -ne "" }
            }

            $cmdStr = $cmd -join " "
            Write-Host "  cmd:     $cmdStr"

            if ($DryRun) {
                Write-Host "  [DRY-RUN] skipping execution" -ForegroundColor Magenta
                continue
            }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                # Stream stdout+stderr to log; tee to console for live progress
                & $cmd[0] $cmd[1..($cmd.Count - 1)] 2>&1 |
                    Tee-Object -FilePath $logFile
                $exitCode = $LASTEXITCODE
            } catch {
                $exitCode = 1
                Write-Error "Run $runIndex failed: $_"
            }
            $sw.Stop()

            if ($exitCode -ne 0) {
                Write-Warning "Run $runIndex (seed=$seed beta-C=$condition) FAILED with exit $exitCode after $($sw.Elapsed)"
                $runs.Add([pscustomobject]@{
                    seed          = $seed
                    beta_c        = $betaC
                    condition     = $condition
                    val_accuracy  = $null
                    exit_code     = $exitCode
                    log_file      = $logFile
                    duration_sec  = [int]$sw.Elapsed.TotalSeconds
                    error         = "training exited with code $exitCode"
                }) | Out-Null
                # T14 is BLOCKING; abort on first failure to surface the issue
                throw "T14 run $runIndex FAILED. Aborting to preserve paired-run integrity. See $logFile"
            }

            # ---------- Extract val_acc from training_history.json ----------
            if (-not (Test-Path $historyFile)) {
                throw "training_history.json not found at $historyFile; cannot extract val_accuracy"
            }
            $history = Get-Content $historyFile -Raw | ConvertFrom-Json
            $lastEpoch = $history | Select-Object -Last 1
            $valAcc = $lastEpoch.val_acc
            if ($null -eq $valAcc) {
                throw "val_acc missing in last epoch entry of $historyFile"
            }

            Write-Host "  val_acc: $valAcc %" -ForegroundColor Green
            Write-Host "  duration: $($sw.Elapsed)"

            $runs.Add([pscustomobject]@{
                seed          = $seed
                beta_c        = $betaC
                condition     = $condition
                val_accuracy  = [double]$valAcc
                exit_code     = 0
                log_file      = $logFile
                duration_sec  = [int]$sw.Elapsed.TotalSeconds
                error         = $null
            }) | Out-Null
        }
    }

    # ---------- Emit t14_results.json ----------
    $results = [pscustomobject]@{
        protocol    = "T14"
        design_ref  = "docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-design.md"
        oracle_ref  = "docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-oracle-report.md"
        dataset     = $Dataset
        epochs      = $Epochs
        seeds       = $Seeds
        run_at      = (Get-Date).ToString("o")
        runs        = $runs
    }
    $outFile = Join-Path $OutputDir "t14_results.json"
    $results | ConvertTo-Json -Depth 6 | Set-Content -Path $outFile -Encoding UTF8
    Write-Host ""
    Write-Host "T14 results written to: $outFile" -ForegroundColor Cyan
    Write-Host "Next: uv run python scripts/analyze_t14_ablation.py --results $outFile" -ForegroundColor Cyan
}
finally {
    Pop-Location
}
