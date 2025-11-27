from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

# 模型 ID
model_id = "qwen/Qwen2.5-7B-Instruct"

# 配置 4-bit 量化（QLoRA 核心）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("正在加载 4-bit 量化模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("✅ 模型加载成功！")
print(f"模型设备: {model.device}")
print(f"显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")