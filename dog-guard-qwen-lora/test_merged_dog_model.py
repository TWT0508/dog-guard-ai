# test_merged_dog_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("🔄 加载基础量化模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    "./model_cache/qwen/Qwen2___5-7B-Instruct",  # 正确的模型路径
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("🔄 合并微调权重...")
finetuned_model = PeftModel.from_pretrained(base_model, "./dog_qwen2.5_qlora")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./dog_qwen2.5_qlora", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("✅ 完整模型加载完成！")

# 测试一个案例
prompt = "### Instruction:\n请根据以下信息生成犬类疾病防治建议：\n\n### Input:\n犬种：金毛寻回犬；地区：上海市；日期：2025-07-15\n\n### Response:\n"

# 修复f-string问题
input_part = prompt.split('### Input:')[1].split('\n')[0]
print(f"\n📋 测试输入: {input_part}")

inputs = tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
with torch.no_grad():
    outputs = finetuned_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_part = response.split("### Response:")[-1].strip()
print(f"🤖 模型回复: {response_part}")