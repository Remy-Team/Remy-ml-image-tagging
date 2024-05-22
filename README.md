# Remy - ML Tagging service 

![demo.gif](docs/figures/demo.gif)

## General info

The repo is based on [SmilingWolf/wd-v1-4-convnext-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2) model.

It is served via BentoML framework. Adaptive batching has been set up.

## API description

Service has a single endpoint `/predict`, which recieves a list of base64 encoded images for inference. 

It is strongly advised to send one image per request, due to batch size BentoML limitations and adaptive batching. Batch requests can cause unexpected behaviour.

```curl
curl -X 'POST' \
  'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'imgs=@04.jpg;type=image/jpeg' 
```

Also, it has infrastructure endpoints for kubernetes endpoints (`/healthz`, `/livez`, `/readyz`) and Prometheus metrics (`/metrics`).

## Build container

Note: only python v3.11.2 has been tested

1. Create build venv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_build.txt
```

2. Fetch model for tagging and save it to bentoml model store.

```bash
cd build_ctx
python save_model.py
```

3. Build container via bentoml

```bash
bentoml build --containerize --version 0.6
```

Output should end with 

```bash
 => exporting to image                                                                                                                                                                0.2s
 => => exporting layers                                                                                                                                                               0.2s
 => => writing image sha256:ab4b4ab06bd34305624b83054f050d4cd6be6ab204cf7506a9fd9393cadf2cb9                                                                                          0.0s
 => => naming to docker.io/library/image_tagging:0.1
```

Note that after building the container you shouuld change version argument to build another container, or delete the previous one.

4. Run container using docker-compose.yml

After building the container, you can start the container with

```bash
docker compose up
```

For your convenience, docker-compose.yml file has been provided to ease deployment purposes. You can adjust environment variables and resources of the service to capabilities of your system there.

```yaml
version: '3.8'
services: 
  image_tagging:
    image: image_tagging:0.1
    ports:
      - 3000:3000
    environment:
      - NUM_WORKERS=5
      - CPUS_PER_WORKER=1
      - BATCH_SIZE=8
    deploy:
      resources:
        limits:
          cpus: '5'
          memory: 10G
```


## FAQ

### How do I start the bentoml service locally without docker to develop?

```bash
cd build_ctx
bentoml serve --reload
```

This will start the service with reloading on `service.py` change.

### Throughput? 

0.65 images per sec per worker on 1 cpu core, around 2gb per worker + 1gb during inference. 

### Test model performance locally? 

Refer to `test_suites/test_wd.py`

```bash
CUDA_VISIBLE_DEVICES="" python test_suites/test_wd.py 
```

If want to use only part of your cpus, for example, to make your application utilize only the first 4 CPUs do:

```bash
CUDA_VISIBLE_DEVICES="" taskset --cpu-list 0-3 python test_suites/test_wd.py 
```

### Something else?

[Ask me](t.me/Quakumei)
