from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import time

# 模型和 tokenizer 路径
model_path = "model/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/model_q4.onnx"
tokenizer_path = "model/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=True,
    local_files_only=True
)

# 创建 ONNX Runtime Session profiling
# so = onnxruntime.SessionOptions()
# so.enable_profiling = True
# session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=so)
session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# 打印模型输入信息
# print("\nModel input info:")
# for input_meta in session.get_inputs():
#     print(f"{input_meta.name}: {input_meta.shape}, type: {input_meta.type}")

# 设置生成参数
max_new_tokens = 256
eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|endoftext|>")
user_token_ids = tokenizer.encode("User:", add_special_tokens=False)

chat_history = ""

while True:
    user_input = input("请输入您的问题（输入 'exit' 退出，'clear' 清空上下文）：")
    if user_input.lower() == 'exit':
        print("结束对话。")
        break
    if user_input.lower() == 'clear':
        print("已清空上下文。")
        chat_history = ""
        continue

    # 构造对话上下文
    chat_history += f"User: {user_input}\nAssistant:"
    prompt = chat_history

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 初始化生成状态
    generated_ids = input_ids.copy()
    past_key_values = {}
    for i in range(28):
        empty_shape = (1, 2, 0, 128)
        past_key_values[f"past_key_values.{i}.key"] = np.zeros(empty_shape, dtype=np.float32)
        past_key_values[f"past_key_values.{i}.value"] = np.zeros(empty_shape, dtype=np.float32)

    start_time = time.time()

    for step in range(max_new_tokens):
        if step == 0:
            input_token = input_ids
            position_ids = np.arange(input_token.shape[1])[None, :].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
        else:
            input_token = next_token_id[:, None]
            position_ids = np.array([[generated_ids.shape[1] - 1]], dtype=np.int64)
            attention_mask = np.ones_like(generated_ids, dtype=np.int64)

        # 构造输入字典
        onnx_inputs = {
            "input_ids": input_token.astype(np.int64),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        onnx_inputs.update(past_key_values)

        outputs = session.run(None, onnx_inputs)

        logits = outputs[0]
        next_token_id = np.argmax(logits[:, -1, :], axis=-1)

        if next_token_id[0] == eos_token_id:
            # print(f"遇到结束符，提前终止生成，生成长度：{generated_ids.shape[1]}")
            break

        generated_ids = np.concatenate([generated_ids, next_token_id[:, None]], axis=1)

        if len(generated_ids[0]) >= len(user_token_ids):
            if list(generated_ids[0][-len(user_token_ids):]) == user_token_ids:
                break

        # 更新 past_key_values
        new_past = {}
        for i in range(28):
            new_past[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            new_past[f"past_key_values.{i}.value"] = outputs[2 + i * 2]
        past_key_values = new_past

    elapsed = time.time() - start_time
    tps = generated_ids.shape[1] / elapsed
    print(f"\nTPS: {tps:.2f} tokens/second, step: ", step)

    # 解码生成结果
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    reply = generated_text[len(prompt):].strip()

    # 清理结尾的 "User:" 和换行符
    if "User:" in reply:
        reply = reply.split("User:")[0].rstrip()

    # 打印响应
    print("\nResponse:")
    print(reply)

    # 添加回答到上下文
    chat_history += f" {reply}\n"

# profiling 结束
# prof_file = session.end_profiling()
# print(f"ONNX Runtime profiling file: {prof_file}")
