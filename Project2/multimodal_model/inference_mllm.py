# multimodal_model/inference_mllm.py

import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from collections import defaultdict

from utils.training_utils import get_device
from vision_transformer.vit import ViT
from language_model.llm import GPTModel
from language_model.tokenizer import CharacterTokenizer
from .connector import Connector
from .mllm import MLLM


# ... (helper functions create_transform and _load_all_captions remain the same) ...
def create_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_all_captions(captions_path: str) -> dict:
    if not os.path.exists(captions_path):
        print(
            f"Warning: Captions file not found at {captions_path}. Cannot display ground-truth."
        )
        return {}
    captions_map = defaultdict(list)
    with open(captions_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                img_file, caption = parts
                captions_map[img_file.strip()].append(caption.strip())
    return captions_map


def inference(config: dict):
    """
    使用训练好的 MLLM 模型为单个图像生成描述。

    [ 任务 ]
    1. 加载训练好的模型权重。
    2. 对输入的 PIL Image 对象进行预处理，使其符合模型输入要求。
    3. 调用 mllm.generate 方法生成描述。
    """
    # --- 0. Setup ---
    device = get_device(config["training"]["device"])
    paths_cfg = config["paths"]
    model_cfg = config["model"]
    infer_cfg = config["generation"]

    # --- 1. Load Ground-Truth Captions (already done) ---
    captions_file_path = paths_cfg["captions_corpus_path"]
    all_gt_captions = _load_all_captions(captions_file_path)

    # --- 2. Load Tokenizer (already done) ---
    tokenizer = CharacterTokenizer(corpus=None)
    tokenizer.load_vocab(paths_cfg["tokenizer_save_path"])
    vocab_size = tokenizer.get_vocab_size()

    # --- 3. Initialize Model Architecture (already done) ---
    vision_encoder = ViT(...).to(device)  # Simplified for brevity
    language_model = GPTModel(...).to(device)  # Simplified for brevity
    connector = Connector(...).to(device)  # Simplified for brevity
    mllm = MLLM(vision_encoder, language_model, connector, tokenizer).to(device)
    print("MLLM architecture built successfully.")

    # --- 4. Load Trained Weights ---
    model_path = paths_cfg["best_model_save_path"]
    if not os.path.exists(model_path):
        print(f"Error: MLLM checkpoint not found at {model_path}")
        return

    # --- START OF STUDENT TASK 1 ---
    # 加载训练好的模型权重。
    # `model_path` 变量已经包含了权重的路径。
    # 提示: 使用 `torch.load()` 加载文件，然后使用模型的 `load_state_dict()` 方法。
    # 确保使用 `map_location=device` 以便在 CPU 或 GPU 上都能正确加载。

    state_dict = torch.load(model_path, map_location=device)
    mllm.load_state_dict(state_dict)

    # --- END OF STUDENT TASK 1 ---

    print(f"Loaded MLLM weights from {model_path}")
    mllm.eval()

    # --- 5. Prepare Input Image ---
    image_path = infer_cfg["inference_image_path"]
    if not os.path.exists(image_path):
        print(f"Error: Inference image not found at {image_path}. Using a dummy image.")
        image = Image.new(
            "RGB",
            (
                model_cfg["vision_encoder"]["image_size"],
                model_cfg["vision_encoder"]["image_size"],
            ),
            color="green",
        )
    else:
        image = Image.open(image_path).convert("RGB")

    transform = create_transform(model_cfg["vision_encoder"]["image_size"])

    # --- START OF STUDENT TASK 2 ---
    # 对 PIL Image 对象 `image` 进行变换，并准备成模型所需的张量格式。
    # 步骤:
    # 1. 使用上面定义的 `transform` 对 `image` 进行处理。
    # 2. 模型期望的输入形状是 [Batch, Channels, Height, Width]，所以需要增加一个批次维度。
    # 3. 将张量移动到正确的设备 (`device`) 上。
    # 提示: 使用 `.unsqueeze(0)` 来增加维度。

    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- END OF STUDENT TASK 2 ---

    prompt = infer_cfg["prompt"]
    image_filename = os.path.basename(image_path)
    ground_truth_for_image = all_gt_captions.get(image_filename, [])

    # --- 6. Generate Caption ---
    print("-" * 50)
    print(f"Image: {image_filename}")
    print("Generating caption...")

    # --- START OF STUDENT TASK 3 ---
    # 调用 `mllm.generate` 方法来生成描述文本。
    #
    # `mllm.generate` 方法的签名是:
    # generate(self, image, prompt, max_new_tokens, temperature, top_k)
    #
    # 请传入以下参数:
    # - image:          你刚刚准备好的 `image_tensor`
    # - prompt:         已经定义好的 `prompt` 变量
    # - max_new_tokens: 从配置 `infer_cfg` 中获取
    # - temperature:    从配置 `infer_cfg` 中获取
    # - top_k:          从配置 `infer_cfg` 中获取 (可能是可选的)
    #
    # 确保在 `torch.no_grad()` 上下文中执行，以节省内存并加速计算。

    with torch.no_grad():
        generated_text = mllm.generate(
            image=image_tensor,
            prompt=prompt,
            max_new_tokens=infer_cfg["max_new_tokens"],
            temperature=infer_cfg["temperature"],
            top_k=infer_cfg.get("top_k", None),
        )
    # --- END OF STUDENT TASK 3 ---

    # --- 7. Print Results ---
    print("-" * 50)
    print(">>> Generated Caption:")
    print(generated_text)
    print("-" * 50)
    if ground_truth_for_image:
        print(">>> Ground-Truth Captions:")
        for i, gt_caption in enumerate(ground_truth_for_image):
            print(f"{i+1}: {gt_caption}")
    else:
        print(">>> No ground-truth captions found for this image.")
    print("-" * 50)
