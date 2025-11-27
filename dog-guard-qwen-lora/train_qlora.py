# train_qlora_minimal.py
import json
from datasets import Dataset
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# ======================
# 配置
# ======================
MODEL_NAME = "qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "dog_prevention.json"
OUTPUT_DIR = "./dog_qwen2.5_qlora"

# LoRA 配置
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 训练参数
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
EPOCHS = 3

# ======================
# 加载数据并预处理
# ======================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 直接构建完整的 prompt 到数据集中
processed_data = []
for item in data:
    text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
    processed_data.append({"text": text})

dataset = Dataset.from_list(processed_data)

# ======================
# 使用魔搭加载 tokenizer 和模型
# ======================
from modelscope import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 准备模型用于训练
model = prepare_model_for_kbit_training(model)

# 添加 LoRA 适配器
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ======================
# SFTTrainer 设置（最小化参数）
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    push_to_hub=False,
    fp16=True,
    report_to=None,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # 数据集已包含完整的 text 字段
    # 移除所有可能冲突的参数
)

# ======================
# 开始训练
# ======================
print("🚀 开始训练...")
trainer.train()

# ======================
# 保存最终模型
# ======================
trainer.save_model(OUTPUT_DIR)
print(f"✅ 模型已保存至: {OUTPUT_DIR}")