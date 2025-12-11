我使用魔搭社区创建的notebook来进行Lora微调

### GPU环境
配置：8核 32GB 显存16G
预装：ModelScope Library
预装镜像：ubuntu22.04-cuda12.4.0-py311-torch2.8.0-1.32.0-LLM

Qwen2.5-7B-Instruct 是一个70亿参数的大模型。notebook显存可能不足,启用量化（4-bit 或 8-bit）+ 梯度检查点。我们可以用 QLoRA（4-bit LoRA） 技术，将显存占用压到 10GB 以内
所需工具：bitsandbytes：支持 4-bit 量化;accelerate + peft：支持 QLoRA 微调

### 安装必要依赖

`pip install -U transformers accelerate peft bitsandbytes datasets einops sentencepiece trl`

### 测试是否能够在环境中成功加载4-bit量化的Qwen2.5-7B-Instruct模型：
`python test_quantized_model.py`

准备好微调的数据：dog_prevention.json

## 开始Lora微调：

### 开始训练
` cd dog-guard-qwen-lora/ `
` python train_qlora.py `

### 输出

```
{'loss': 1.2643, 'grad_norm': 1.180071234703064, 'learning_rate': 0.00017777777777777779, 'entropy': 1.1745547741651534, 'num_tokens': 9072.0, 'mean_token_accuracy': 0.722261368483305, 'epoch': 0.38}
{'loss': 0.6754, 'grad_norm': 0.9385711550712585, 'learning_rate': 0.0001530864197530864, 'entropy': 0.6852330416440964, 'num_tokens': 18248.0, 'mean_token_accuracy': 0.819330146163702, 'epoch': 0.75}
{'loss': 0.5353, 'grad_norm': 0.749603807926178, 'learning_rate': 0.00012839506172839505, 'entropy': 0.5842538946553281, 'num_tokens': 26976.0, 'mean_token_accuracy': 0.8563290207009566, 'epoch': 1.11}
{'loss': 0.3313, 'grad_norm': 0.7676432132720947, 'learning_rate': 0.0001037037037037037, 'entropy': 0.3603133101016283, 'num_tokens': 36075.0, 'mean_token_accuracy': 0.8976029336452485, 'epoch': 1.49}
{'loss': 0.3309, 'grad_norm': 0.8327499032020569, 'learning_rate': 7.901234567901235e-05, 'entropy': 0.38275447860360146, 'num_tokens': 45205.0, 'mean_token_accuracy': 0.8997883692383766, 'epoch': 1.87}
{'loss': 0.2504, 'grad_norm': 0.607859194278717, 'learning_rate': 5.4320987654320986e-05, 'entropy': 0.3392490256381662, 'num_tokens': 53881.0, 'mean_token_accuracy': 0.9238561289875131, 'epoch': 2.23}
{'loss': 0.2017, 'grad_norm': 0.6754600405693054, 'learning_rate': 2.962962962962963e-05, 'entropy': 0.2621204825118184, 'num_tokens': 63005.0, 'mean_token_accuracy': 0.935276598483324, 'epoch': 2.6}
{'loss': 0.2004, 'grad_norm': 0.7405008673667908, 'learning_rate': 4.938271604938272e-06, 'entropy': 0.2626534206792712, 'num_tokens': 72176.0, 'mean_token_accuracy': 0.9368092328310013, 'epoch': 2.98}
{'train_runtime': 791.4845, 'train_samples_per_second': 0.804, 'train_steps_per_second': 0.102, 'train_loss': 0.4698978103237388, 'entropy': 0.25272684171795845, 'num_tokens': 72618.0, 'mean_token_accuracy': 0.9358385503292084, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [13:11<00:00,  9.77s/it]
✅ 模型已保存至: ./dog_qwen2.5_qlora
```
### 合并模型权重
将微调后的模型与基础模型（Qwen2.5-7B-Instruct）权重合并，先把模型下载到当前目录的 model_cache 文件夹中,

```
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./model_cache')
print(f'模型已下载至: {model_dir}')
"
```

### 合并权重并测试微调后的模型效果
`python test_merged_dog_model.py`
