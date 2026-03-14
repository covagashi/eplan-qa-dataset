import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LORA_ID = "covaga/eplan-assistant-v2-lora"

SYSTEM_MSG = """You are an expert EPLAN Electric P8 assistant specialized in industrial electrical engineering.
You help engineers with API usage, troubleshooting, best practices, and procedural guidance.
You can write complete, working C# scripts using the EPLAN API.
Provide accurate, detailed answers based on EPLAN documentation."""

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, LORA_ID)
model = model.merge_and_unload()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
print("Model loaded!")


def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
    )
    response = outputs[0]["generated_text"][len(prompt):].strip()
    return response


demo = gr.ChatInterface(
    fn=respond,
    title="EPLAN Electric P8 Assistant",
    description="Fine-tuned Qwen 2.5 3B on 5,116 EPLAN Q&A pairs. Ask about API usage, troubleshooting, scripting, and best practices.",
    examples=[
        "How do I export a project to PDF using the EPLAN API?",
        "What is the difference between a Function and a FunctionBase?",
        "Write a C# script that iterates all pages in an EPLAN project",
        "My script throws NullReferenceException when iterating pages. What could cause this?",
    ],
    theme="soft",
)

demo.launch()
