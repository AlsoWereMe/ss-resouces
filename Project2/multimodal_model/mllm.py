# multimodal_model/mllm.py

import torch
import torch.nn as nn
from vision_transformer.vit import ViT
from language_model.llm import GPTModel 
from .connector import Connector

class MLLM(nn.Module):
    """
    多模态大模型 (Multimodal Large Language Model)。
    """
    def __init__(self, vision_encoder: ViT, language_model: GPTModel, connector: Connector, tokenizer):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.connector = connector
        self.tokenizer = tokenizer

    def freeze_parameters(self, freeze_vit: bool, freeze_llm: bool):
        # 此函数无需修改
        if freeze_vit:
            for param in self.vision_encoder.parameters(): param.requires_grad = False
            print("ViT parameters frozen.")
        else:
            for param in self.vision_encoder.parameters(): param.requires_grad = True
            print("ViT parameters are trainable.")
        if freeze_llm:
            for param in self.language_model.parameters(): param.requires_grad = False
            print("LLM parameters frozen.")
        else:
            for param in self.language_model.parameters(): param.requires_grad = True
            print("LLM parameters are trainable.")

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor):
        """
        用于训练的前向传播。
        """
        # --- START OF STUDENT MODIFICATION (FORWARD) ---
        
        # 1. 从图像中提取视觉特征。
        #    - 使用 `self.vision_encoder.forward_features()`。
        visual_features = self.vision_encoder.forward_features(images)   # [B, Nv, Dv]

        # 2. 使用 Connector 将视觉特征投影到语言模型的嵌入空间。
        visual_embeddings = self.connector(visual_features)
        
        # 3. 获取文本的嵌入。
        #    - 使用 `self.language_model.token_embedding()`。
        text_embeddings = self.language_model.token_embedding(text_tokens)

        # 4. [关键步骤] 拼接视觉嵌入和文本嵌入。
        #    - 这是多模态融合发生的地方。
        #    - 视觉嵌入应该在文本嵌入之前。
        #    - 使用 `torch.cat` 并在 `dim=1` (序列维度) 上拼接。
        inputs_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
        
        # 5. 将拼接后的嵌入传入 LLM。
        #    - 使用 `self.language_model.forward_from_embeddings()`。
        logits = self.language_model.forward_from_embeddings(inputs_embeddings)
        
        # --- END OF STUDENT MODIFICATION (FORWARD) ---
        
        return logits, visual_embeddings.size(1)

    @torch.no_grad()
    def generate(self, image: torch.Tensor, prompt: str, max_new_tokens: int, temperature: float = 0.7, top_k: int = None):
        """
        用于推理的自回归文本生成。
        """
        self.eval()
        device = image.device
        sos_token_id = self.tokenizer.get_sos_token_id()
        eos_token_id = self.tokenizer.get_eos_token_id()
        
        # --- START OF STUDENT MODIFICATION (GENERATE) ---

        # 1. 编码图像 (已完成)
        visual_features = self.vision_encoder.forward_features(image)
        visual_embeddings = self.connector(visual_features) # Shape: [1, N_visual, D_lang]

        # 2. 构建初始文本序列 (已完成)
        initial_tokens = [sos_token_id]
        if prompt:
            initial_tokens.extend(self.tokenizer.encode(prompt))
        prompt_tokens = torch.tensor(initial_tokens, dtype=torch.long, device=device).unsqueeze(0)
        prompt_embeddings = self.language_model.token_embedding(prompt_tokens) # Shape: [1, N_prompt, D_lang]

        # 3. 拼接视觉和文本嵌入作为初始输入 (已完成)
        input_embeddings = torch.cat([visual_embeddings, prompt_embeddings], dim=1)
        
        generated_tokens = []
        
        # 4. 自回归生成循环
        for _ in range(max_new_tokens):
            # a. 从当前的 `input_embeddings` 获取 logits。
            #    - 使用 `self.language_model.forward_from_embeddings()`。
            logits = self.language_model.forward_from_embeddings(input_embeddings) 
            
            # b. 只取序列中最后一个时间步的 logits，因为我们只关心预测下一个 token。
            #    - `logits` 的形状是 (B, T, C)，我们需要形状为 (B, C) 的部分。
            #    - 提示: `logits[:, -1, :]`
            next_token_logits = logits[:, -1, :] 
            
            # (Top-k 和 temperature 缩放已完成)
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            next_token_logits = next_token_logits / temperature
            
            # c. 将 logits 转换为概率分布。
            #    - 使用 `torch.softmax`。
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # d. 从概率分布中采样一个 token。
            #    - 使用 `torch.multinomial`。
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            if next_token.item() == eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            
            # TODO: e. [关键步骤] 为下一次迭代准备输入。
            #    - 获取新生成的 `next_token` 的嵌入。
            #    - 将这个新的嵌入拼接到当前的 `input_embeddings` 的末尾。
            #    - 这比每次都重新编码整个序列要高效得多。
            next_token_embedding = self.language_model.token_embedding(next_token)
            input_embeddings = torch.cat([input_embeddings, next_token_embedding], dim=1)

            if input_embeddings.size(1) >= self.language_model.max_len:
                break
        
        # --- END OF STUDENT MODIFICATION (GENERATE) ---

        self.train()
        return self.tokenizer.decode(generated_tokens)