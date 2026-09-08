# miniGPT-server (V1)

## Local install
pip install -r requirements.txt

## Run locally
python app.py

## Health check
curl http://localhost:5000/

## Generate (streams text)
curl.exe -X POST http://localhost:5000/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A wise crow\",\"max_new_tokens\":50,\"temperature\":0.7,\"top_p\":0.9}"

## Dev diagnostics
curl http://localhost:5000/dev/health
curl http://localhost:5000/dev/stats
curl.exe -X POST http://localhost:5000/dev/generate -H "Content-Type: application/json" -d "{\"prompt\":\"A wise crow\"}"
curl.exe -X POST http://localhost:5000/dev/evaluate -H "Content-Type: application/json" -d "{\"prompt\":\"The cat sat on\",\"target\":\"the mat\"}"

## Docker build
docker build -t minigpt-server .

## Docker run
docker run -p 5000:5000 minigpt-server

## Docker curl test
curl http://localhost:5000/

## Note: The complete testing and publishing needs to be done...

## hugging face link for the model files: 
https://huggingface.co/ARX1A07/miniGPT_Project
