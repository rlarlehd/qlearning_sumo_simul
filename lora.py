# server.py (FastAPI 기반 학습 및 추론 서버)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
import uuid
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

app = FastAPI()

# 데이터 구조 정의
class TrainSample(BaseModel):
    input: str
    output: str

class TrainRequest(BaseModel):
    data: List[TrainSample]
    epochs: Optional[int] = 3

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

MODEL_DIR = "./latest_trained_model"
TOKENIZER = None
MODEL = None

@app.post("/train")
async def train_model(request: TrainRequest):
    data = request.data
    epochs = request.epochs

    if not data:
        raise HTTPException(status_code=400, detail="No training data provided.")

    dataset = Dataset.from_list([sample.dict() for sample in data])

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=False
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)

    def preprocess(example):
        prompt = example["input"]
        response = example["output"]
        full_input = f"{prompt}\n### 답변:\n"
        full_text = full_input + response

        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = len(tokenizer(full_input, return_tensors="pt")["input_ids"][0])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_dataset = dataset.map(preprocess)

    output_dir = MODEL_DIR
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        learning_rate=5e-5,
        logging_steps=10,
        output_dir=output_dir,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    global TOKENIZER, MODEL
    TOKENIZER = tokenizer
    MODEL = model

    return {"message": "Training complete", "model_path": output_dir, "epochs": epochs}

@app.post("/infer")
async def infer(request: InferenceRequest):
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                device_map="auto",
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            )
            MODEL = PeftModel.from_pretrained(base_model, MODEL_DIR).to("cuda")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}")

    input_text = f"{request.prompt}\n### 답변:\n"
    inputs = TOKENIZER(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    result = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    answer = result.split("### 답변:")[-1].strip()
    # 학습되지 않은 내용에 대한 검증 로직 추가
    if "학습되지 않은 내용" in answer or answer.strip() == "":
        return {"response": "죄송합니다. 해당 내용에 대해 학습된 정보가 없어 답변할 수 없습니다."}

    return {"response": answer}

# 실행 명령:
# uvicorn apiServer:app --host 0.0.0.0 --port 8000