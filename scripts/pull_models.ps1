$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$modelsRoot = Join-Path $projectRoot "models"
New-Item -ItemType Directory -Force -Path $modelsRoot | Out-Null

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw ".venv python bulunamadi: $venvPython"
}

$script = @'
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path("models")
root.mkdir(parents=True, exist_ok=True)

models = {
    "newmindai/Mursit-Large-TR-Retrieval": "newmindai__Mursit-Large-TR-Retrieval",
    "BAAI/bge-reranker-v2-m3": "BAAI__bge-reranker-v2-m3",
    "newmindai/ettin-encoder-150M-TR-HD": "newmindai__ettin-encoder-150M-TR-HD",
}

for repo_id, folder in models.items():
    local_dir = root / folder
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

print("Model indirme tamamlandi.")
'@

Push-Location $projectRoot
try {
    $script | & $venvPython -
}
finally {
    Pop-Location
}
