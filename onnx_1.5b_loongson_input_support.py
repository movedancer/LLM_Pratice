from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import time

# 模型和tokenizer路径
model_path = "model/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/model_q4.onnx"
tokenizer_path = "model/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=True,
    local_files_only=True
)

# 开启ort内置profiling
so = onnxruntime.SessionOptions()
so.enable_profiling = True  # 开启profiling

# 创建ONNX Session并打印输入信息
session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options = so)
print("\nModel input info:")
for input_meta in session.get_inputs():
    name = input_meta.name
    shape = input_meta.shape
    dtype = input_meta.type
    print(f"{name}: {shape}, type: {dtype}")

# 初始化生成参数
max_new_tokens = 256
eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|endoftext|>")

# 维护历史队列
chat_history = ""

while True:
    # 用户输入
    user_input = input("请输入您的问题（输入'exit'结束），'clear' 清空上下文）：")
    if user_input.lower() == 'exit':
        print("结束对话。")
        break
    if user_input.lower() == 'clear':
        print("已清空上下文。")
        chat_history = ""
        continue
    # 构建prompt
    prompt = chat_history + f"User: {user_input}\nAssistant:"

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = np.arange(input_ids.shape[1])[None, :].astype(np.int64)
    generated_ids = input_ids.copy()
    # 构造空past_key_values
    past_key_values = {}
    for i in range(28):
        shape = (1, 2, 0, 128)  # 根据输出打印的模型输入要求
        past_key_values[f"past_key_values.{i}.key"] = np.zeros(shape, dtype=np.float32)
        past_key_values[f"past_key_values.{i}.value"] = np.zeros(shape, dtype=np.float32)

    # 构造ONNX输入
    onnx_inputs = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask.astype(np.int64),
        "position_ids": position_ids.astype(np.int64),
    }
    onnx_inputs.update(past_key_values)

    start_time = time.time()  # 计时开始

    for step in range(max_new_tokens):
        position_ids = np.arange(generated_ids.shape[1])[None, :].astype(np.int64)
        attention_mask = np.ones_like(generated_ids, dtype=np.int64)

        past_key_values = {}
        for i in range(28):
            shape = (1, 2, 0, 128)
            past_key_values[f"past_key_values.{i}.key"] = np.zeros(shape, dtype=np.float32)
            past_key_values[f"past_key_values.{i}.value"] = np.zeros(shape, dtype=np.float32)

        onnx_inputs = {
            "input_ids": generated_ids.astype(np.int64),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        onnx_inputs.update(past_key_values)

        start_op_time = time.time()
        outputs = session.run(None, onnx_inputs)
        end_op_time = time.time()
        print(f"session.run() 耗时: {end_op_time - start_op_time:.6f} 秒")
        logits = outputs[0]

        next_token_id = np.argmax(logits[:, -1, :], axis=-1)

        print("[1]", repr(tokenizer.decode(generated_ids[0], skip_special_tokens=False)))
        generated_ids = np.concatenate([generated_ids, next_token_id[:, None]], axis=1)
        print("[2]", repr(tokenizer.decode(generated_ids[0], skip_special_tokens=False)))
        if next_token_id[0] == eos_token_id:
            print(f"遇到结束符，提前终止生成，生成长度：{generated_ids.shape[1]}")
            break
    # 计算TPS
    elapsed = time.time() - start_time
    tps = generated_ids.shape[1] / elapsed
    print(f"\nTPS: {tps:.2f} tokens/second")

    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    reply = generated_text[len(prompt):]
    chat_history += f"User: {user_input}\nAssistant: {reply}"

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nResponse:")
    print(generated_text[len(prompt):].strip())

prof_file = session.end_profiling()
print(f"ONNX Runtime profiling file: {prof_file}")