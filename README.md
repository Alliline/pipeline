# Pipeline

Создать/загрузить:
1. images - папку с картинками
2. dataset.jsonl - массив вида {id: int, file: string (local path)}

Картинки заранее сжаты до 768px по наибольшей стороне, чтобы не создавать тормоза сверху.
Цвета лучше получить в другом месте, как-нибудь подешевле. Иначе случается доп. вызов CLIP'a.
```
convert {source} -resize '768x768>' -quality 90 {target}
```

Запускать и ждать (tmux):
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
export INPUT_FILE="dataset.jsonl"
export OUTPUT_FILE="results.jsonl"
export NUM_GPUS=4
export DTYPE=bf16
export BATCH_SIZE_PER_GPU=12
export MAX_NEW_TOKENS=96
export RESIZE_IN_PYTHON=False
export MAX_PIXELS=351232

python pipelinev8.py
-- или --
python pipelinev8.py > pipe.log 2>&1
```

Чекать прогресс:
```
echo "$(date +%T) $(wc -l .__tmp_results_gpu*.jsonl | tail -n 1)"
```

Дополнительно (для моего случая):
```
apt install mc rclone htop tmux
pip install torch torchvision transformers scikit-learn pillow tqdm accelerate
cd /dev/shm/
mkdir -p /dev/shm/hf_cache
mkdir -p /dev/shm/tmp
export HF_HOME=/dev/shm/hf_cache
export HUGGINGFACE_HUB_CACHE=/dev/shm/hf_cache/hub
export TRANSFORMERS_CACHE=/dev/shm/hf_cache/hub
export TMPDIR=/dev/shm/tmp
```
