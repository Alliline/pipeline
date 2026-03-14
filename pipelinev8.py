# Pipeline
# @alliline ; @ooofrieren
# + GPT5.4, Gemini, Deepseek, Qwen
import os
import gc
import json
import time
import inspect
import traceback
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# =========================================================
# HIGH-THROUGHPUT MULTI-GPU IMAGE ANALYSIS PIPELINE FOR 4xH200
# =========================================================
# Key changes vs pipelinev6:
# 1) Migrated from Qwen2.5-VL to Qwen3-VL.
# 2) Fully removed color recognition and CLIP dependency.
# 3) Reworked execution model from one sharded model to 4 independent GPU workers.
#    This improves throughput on H200 by using data parallel inference instead of
#    one giant cross-GPU generation job.
# 4) Added bounded visual token budget through processor.max_pixels.
# 5) Safer JSON-only prompting + trimmed decoding.
# 6) Separate shard outputs per GPU, then deterministic merge at the end.
# =========================================================

# -----------------------------
# USER CONFIG
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-32B-Instruct")
INPUT_FILE = os.getenv("INPUT_FILE", "dataset.jsonl")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "results.jsonl")

# If your images are already pre-resized offline, keep this False.
# Recommended: resize offline to long side 768 px and keep aspect ratio.
RESIZE_IN_PYTHON = False
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "768"))

# Number of parallel model replicas.
# For 4xH200 it is usually best to run one replica per GPU.
NUM_GPUS = int(os.getenv("NUM_GPUS", str(torch.cuda.device_count())))

# Per-GPU microbatch. Start here, then raise until throughput plateaus.
# For Qwen3-VL-32B-Instruct on H200, 12-24 is a realistic starting range
# depending on prompt length and max_new_tokens.
BATCH_SIZE_PER_GPU = int(os.getenv("BATCH_SIZE_PER_GPU", "12"))

# CPU image loading threads per GPU worker.
LOAD_THREADS = int(os.getenv("LOAD_THREADS", "8"))

# Generation limit. Lower = faster.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))

# Qwen processor visual budget.
# 448 * 28 * 28 ~= 351k input pixels, close to ~768x448 budget.
# Good speed/quality tradeoff for <= FullHD inputs.
MIN_PIXELS = int(os.getenv("MIN_PIXELS", str(224 * 224)))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(448 * 28 * 28)))

# bf16 is preferred on H200.
DTYPE_STR = os.getenv("DTYPE", "bfloat16").lower()

# Use FA2 if installed, else SDPA.
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# TORCH / CUDA TUNING
# -----------------------------

def configure_torch() -> None:
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


def get_dtype() -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if DTYPE_STR not in mapping:
        raise ValueError(f"Unsupported DTYPE={DTYPE_STR}")
    return mapping[DTYPE_STR]


DTYPE = get_dtype()

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = """Analyze the image and return STRICT JSON only.

Schema:
{
  "theme": [string],
  "style": [string],
  "tags": [string],
  "characters": [string],
  "objects": [string],
  "scene": string,
  "location": string,
  "description": string,
  "content_flags": {
    "nsfw": boolean,
    "loli": boolean,
    "violence": boolean,
    "gore": boolean,
    "nazi": boolean,
    "extremism": boolean,
    "lgbt": boolean,
    "disturbing": boolean,
    "scary": boolean,
    "propaganda": boolean,
    "drugs": boolean
  }
}

Rules:
- Return JSON only.
- No markdown.
- No explanations.
- Use empty arrays or empty strings when unknown.
- Keep description concise and factual.
"""


def build_messages() -> List[Dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }]


# -----------------------------
# DATASET HELPERS
# -----------------------------

def load_processed_ids(path: str) -> set:
    processed = set()
    if not os.path.exists(path):
        return processed

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    processed.add(str(obj["id"]))
            except json.JSONDecodeError:
                continue
    return processed


def dataset_iter(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def shard_dataset(items: List[Dict], num_shards: int) -> List[List[Dict]]:
    shards = [[] for _ in range(num_shards)]
    for idx, item in enumerate(items):
        shards[idx % num_shards].append(item)
    return shards


# -----------------------------
# IMAGE LOADING
# -----------------------------

def maybe_resize(img: Image.Image) -> Image.Image:
    if not RESIZE_IN_PYTHON:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_SIDE:
        return img
    scale = MAX_IMAGE_SIDE / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.BICUBIC)


def load_image(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = maybe_resize(img)
            img.load()
            return img
    except Exception:
        return None


def load_images_parallel(batch: List[Dict]) -> Tuple[List[Dict], List[Image.Image]]:
    with ThreadPoolExecutor(max_workers=LOAD_THREADS) as ex:
        imgs = list(ex.map(lambda x: load_image(x["file"]), batch))

    filtered_items, filtered_imgs = [], []
    for item, img in zip(batch, imgs):
        if img is not None:
            filtered_items.append(item)
            filtered_imgs.append(img)
    return filtered_items, filtered_imgs


# -----------------------------
# MODEL LOADING
# -----------------------------

def load_model_for_gpu(model_name: str, device: str):
    # Compatibility shim because some model examples/documentation use dtype,
    # while older transformers loaders still expect torch_dtype.
    loader = Qwen3VLForConditionalGeneration.from_pretrained
    sig = inspect.signature(loader)

    kwargs = {
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map": {"": device},
    }

    if "dtype" in sig.parameters:
        kwargs["dtype"] = DTYPE
    else:
        kwargs["torch_dtype"] = DTYPE

    try:
        model = loader(model_name, **kwargs)
    except TypeError:
        # Fallback for mixed environments.
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = DTYPE
        model = loader(model_name, **kwargs)

    model.eval()
    return model


# -----------------------------
# INFERENCE
# -----------------------------

def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start:end + 1]
    return text.strip()


def analyze_batch(model, processor, chat_template: str, images: List[Image.Image]) -> List[str]:
    texts = [chat_template] * len(images)

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)
    device = next(model.parameters()).device
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens; avoids prompt echo and speeds parsing.
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded


# -----------------------------
# WORKER PROCESS
# -----------------------------

def worker_main(rank: int, items: List[Dict], output_path: str):
    configure_torch()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    started = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    processor.tokenizer.padding_side = "left"

    messages = build_messages()
    chat_template = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model = load_model_for_gpu(MODEL_NAME, device)

    written = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for start_idx in range(0, len(items), BATCH_SIZE_PER_GPU):
            batch = items[start_idx:start_idx + BATCH_SIZE_PER_GPU]
            batch_items, batch_images = load_images_parallel(batch)
            if not batch_items:
                failed += len(batch)
                continue

            try:
                results = analyze_batch(model, processor, chat_template, batch_images)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                # Fallback: split the microbatch once and continue.
                if len(batch_items) == 1:
                    failed += 1
                    continue

                mid = len(batch_items) // 2
                sub_batches = [
                    (batch_items[:mid], batch_images[:mid]),
                    (batch_items[mid:], batch_images[mid:]),
                ]
                results = []
                merged_items = []
                for sub_items, sub_images in sub_batches:
                    if not sub_items:
                        continue
                    sub_res = analyze_batch(model, processor, chat_template, sub_images)
                    results.extend(sub_res)
                    merged_items.extend(sub_items)
                batch_items = merged_items
            except Exception:
                traceback.print_exc()
                failed += len(batch_items)
                continue

            for item, result in zip(batch_items, results):
                record = {
                    "id": item["id"],
                    "analysis": extract_json(result),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            out.flush()

            # Keep allocator healthy during very long runs.
            del batch_items, batch_images, results
            if (start_idx // BATCH_SIZE_PER_GPU) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.time() - started
    print(f"[GPU {rank}] done | written={written} failed={failed} elapsed={elapsed:.1f}s")


# -----------------------------
# MERGE OUTPUTS
# -----------------------------

def merge_shards(shard_paths: List[str], final_output: str):
    with open(final_output, "a", encoding="utf-8") as dst:
        for shard in shard_paths:
            if not os.path.exists(shard):
                continue
            with open(shard, "r", encoding="utf-8") as src:
                for line in src:
                    dst.write(line)


# -----------------------------
# MAIN
# -----------------------------

def main():
    if NUM_GPUS < 1:
        raise RuntimeError("CUDA GPU not found")

    configure_torch()

    print(f"Model: {MODEL_NAME}")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Per-GPU batch size: {BATCH_SIZE_PER_GPU}")
    print(f"DTYPE: {DTYPE}")
    print(f"Visual budget: min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}")
    print(f"Python resize: {RESIZE_IN_PYTHON}, MAX_IMAGE_SIDE={MAX_IMAGE_SIDE}")

    processed_ids = load_processed_ids(OUTPUT_FILE)
    print(f"Found {len(processed_ids)} already processed ids")

    items = []
    skipped = 0
    for item in dataset_iter(INPUT_FILE):
        item_id = str(item.get("id"))
        if item_id in processed_ids:
            skipped += 1
            continue
        items.append(item)

    print(f"Queued {len(items)} new items; skipped {skipped}")
    if not items:
        print("Nothing to process")
        return

    active_gpus = min(NUM_GPUS, len(items))
    shards = shard_dataset(items, active_gpus)

    base_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE)) or "."
    shard_paths = [os.path.join(base_dir, f".__tmp_results_gpu{i}.jsonl") for i in range(active_gpus)]

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(active_gpus):
        p = ctx.Process(target=worker_main, args=(rank, shards[rank], shard_paths[rank]), daemon=False)
        p.start()
        procs.append(p)

    bad = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            bad = True

    if bad:
        raise RuntimeError("One or more GPU workers exited with non-zero status")

    merge_shards(shard_paths, OUTPUT_FILE)
    for path in shard_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    print("Processing finished")


if __name__ == "__main__":
    main()
