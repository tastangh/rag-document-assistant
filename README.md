# rag-document-assistant
rag-document-assistant

FİŞ OCR İÇİN
cd /home/mtastan/workspace/rag-document-assistant
source .venv/bin/activate
pip install -U "huggingface_hub[cli]" datasets
hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir data/cord-v2

