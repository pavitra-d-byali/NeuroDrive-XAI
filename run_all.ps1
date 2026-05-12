# NeuroDrive-XAI — One-Command Runner
# Run from the project root: .\run_all.ps1

param(
    [switch]$SkipInstall,
    [switch]$SkipDataset,
    [switch]$SkipTrain,
    [switch]$SkipPipeline,
    [switch]$SkipTests,
    [switch]$CarlaDemo,
    [string]$Input = "demo/messy_drive.mp4",
    [string]$Output = "artifacts/output_demo.mp4"
)

$ErrorActionPreference = "Stop"

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
    Write-Host ("-" * 55) -ForegroundColor DarkGray
}

function Write-OK($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-SKIP($msg) { Write-Host "  --  $msg (skipped)" -ForegroundColor DarkGray }
function Write-WARN($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  NeuroDrive-XAI - Full Pipeline Runner" -ForegroundColor White
Write-Host ("=" * 60) -ForegroundColor Cyan

# ─────────────────────────── Step 1: Install dependencies ────────────────────
Write-Step "1/7" "Installing Python dependencies"
if (-not $SkipInstall) {
    pip install -r requirements.txt --quiet
    Write-OK "Dependencies installed"
} else { Write-SKIP "Install" }

# ─────────────────────────── Step 2: Export ONNX models ──────────────────────
Write-Step "2/7" "Exporting ONNX models (HybridNets)"
if (Test-Path "weights/hybridnets.onnx") {
    Write-OK "ONNX model already exists (weights/hybridnets.onnx)"
} elseif (Test-Path "weights/hybridnets.pth") {
    Write-Host "  Running setup_models.py..."
    python setup_models.py
    if ($LASTEXITCODE -eq 0) { Write-OK "ONNX export complete" }
    else { Write-WARN "ONNX export failed — will use PyTorch fallback" }
} else {
    Write-WARN "No HybridNets weights found. Download from: https://github.com/datvuthanh/HybridNets"
}

# ─────────────────────────── Step 3: BDD100K dataset ─────────────────────────
Write-Step "3/7" "Setting up BDD100K dataset"
if (-not $SkipDataset) {
    if (Test-Path "datasets/bdd100k/images") {
        Write-OK "BDD100K images found at datasets/bdd100k/"
    } else {
        Write-Host "  Downloading/creating BDD100K sample..."
        python dataset/download_bdd.py --synthetic
        Write-OK "Synthetic BDD100K sample created"
    }

    # Generate real features
    if (Test-Path "dataset/real_features.csv") {
        $rows = (Get-Content "dataset/real_features.csv" | Measure-Object -Line).Lines
        Write-OK "real_features.csv exists ($rows rows)"
    } else {
        Write-Host "  Extracting features from BDD100K images..."
        python dataset/generate_features.py --no-perc --max 1000
        Write-OK "Feature extraction complete"
    }
} else { Write-SKIP "Dataset" }

# ─────────────────────────── Step 4: Train MLP ───────────────────────────────
Write-Step "4/7" "Training NeuroDrive decision MLP"
if (-not $SkipTrain) {
    if (Test-Path "weights/neurodrive_mlp.pth") {
        Write-OK "MLP weights found (weights/neurodrive_mlp.pth)"
        Write-Host "  To retrain: python decision/train.py"
    } else {
        Write-Host "  Training on real features (or synthetic fallback)..."
        python decision/train.py --epochs 30
        if ($LASTEXITCODE -eq 0) { Write-OK "Training complete" }
        else { Write-WARN "Training failed" }
    }
} else { Write-SKIP "Training" }

# ─────────────────────────── Step 5: Run main pipeline ───────────────────────
Write-Step "5/7" "Running main inference pipeline on video"
if (-not $SkipPipeline) {
    if (Test-Path $Input) {
        Write-Host "  Input : $Input"
        Write-Host "  Output: $Output"
        New-Item -ItemType Directory -Force -Path artifacts | Out-Null
        python main_pipeline.py --input $Input --output $Output
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Pipeline complete → $Output"
            Write-OK "XAI log        → artifacts/explanations.json"
        } else {
            Write-WARN "Pipeline failed (check above for errors)"
        }
    } else {
        Write-WARN "Input video not found: $Input"
        Write-Host "  Place a video at demo/messy_drive.mp4 and re-run."
    }
} else { Write-SKIP "Pipeline" }

# ─────────────────────────── Step 6: Run tests ───────────────────────────────
Write-Step "6/7" "Running unit tests"
if (-not $SkipTests) {
    python -m pytest tests/ -v --tb=short 2>&1
    if ($LASTEXITCODE -eq 0) { Write-OK "All tests passed" }
    else { Write-WARN "Some tests failed (see above)" }
} else { Write-SKIP "Tests" }

# ─────────────────────────── Step 7: CARLA demo / Dashboard ──────────────────
Write-Step "7/7" "Launching Dashboard + CARLA demo"

# CARLA-free replay demo
if ($CarlaDemo) {
    Write-Host "  Generating CARLA replay demo..."
    python carla_replay.py --demo
    Write-OK "Replay plots saved → artifacts/replay/"
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  NeuroDrive-XAI — Setup Complete!" -ForegroundColor White
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host ""
Write-Host "  Launch dashboard:" -ForegroundColor Cyan
Write-Host "    streamlit run frontend/app.py" -ForegroundColor White
Write-Host ""
Write-Host "  Launch API:" -ForegroundColor Cyan
Write-Host "    uvicorn api.routes:app --reload --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "  Run CARLA replay (no CARLA needed):" -ForegroundColor Cyan
Write-Host "    python carla_replay.py --demo" -ForegroundColor White
Write-Host ""
Write-Host "  Run CARLA simulation (CARLA required):" -ForegroundColor Cyan
Write-Host "    python carla_run.py --map Town03 --episodes 3" -ForegroundColor White
Write-Host ""
