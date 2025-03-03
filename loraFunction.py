import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>{}"""

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""


# 迭代训练集数据，处理prompt
def formatting_prompts_func(examples, tokenizer):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + tokenizer.eos_token
        texts.append(text)
    return {
        "text": texts,
    }

def formatting_prompts_func_openmind(examples, tokenizer, max_seq_length):
    input_ids, attention_mask, labels = [], [], []

    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]

    texts = tokenizer.bos_token + train_prompt_style.format(inputs, cots, outputs) + tokenizer.eos_token

    input_tokenizer = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids += input_tokenizer['input_ids']
    attention_mask += input_tokenizer['attention_mask']

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids
    }

class DataCollatorForSeq2SeqCustom:
    def __init__(self, tokenizer, padding=True, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.padding = padding  # 是否填充到最大长度
        self.return_tensors = return_tensors  # 返回格式，默认为 pytorch tensor

    def __call__(self, batch):
        # 从 batch 中提取 input_ids, attention_mask, 和 labels
        input_ids = [example['input_ids'] for example in batch]
        attention_mask = [example['attention_mask'] for example in batch]
        labels = [example['labels'] for example in batch]

        # 填充所有 sequences 到最大长度
        input_ids = self.pad_sequence(input_ids)
        attention_mask = self.pad_sequence(attention_mask)
        labels = self.pad_sequence(labels)

        # 如果需要返回 pytorch tensor，则将数据转换为 tensor 格式
        if self.return_tensors == "pt":
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def pad_sequence(self, sequences):
        # 填充序列到最大长度
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [self.tokenizer.pad_token_id] * (max_length - len(seq)) for seq in sequences]
        return padded_sequences

class DeepSeekQLora:
    def __init__(self, args):
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.args = args
        self.max_seq_length = 2048

    def loadModel(self, model_path):
        # 加载模型
        if self.args.fine_tune_type == "unsloth":
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_path,
                max_seq_length = self.max_seq_length,
                dtype = None,
                load_in_4bit = True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, device_map=None)
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)
        return model, tokenizer

    def deepSeekInfer(self, model, tokenizer, question):
        # 推理测试
        model_inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
        if self.args.fine_tune_type == "unsloth":
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=self.max_seq_length,
            use_cache=True,
        )
        response = tokenizer.batch_decode(outputs)
        print(response[0].split("### Response:")[1])

    def dataLoad(self, dataPath):
        if self.args.fine_tune_type == "unsloth":
            dataset = load_dataset(self.args.data_type, data_files=dataPath, split="train")
            self.dataset = dataset.map(formatting_prompts_func,
                                       fn_kwargs={"tokenizer": self.tokenizer},
                                       batched=True)
        else:
            dataset = load_dataset(self.args.data_type, data_files=dataPath, split="train")
            self.dataset = dataset.map(formatting_prompts_func_openmind,
                                       fn_kwargs={"tokenizer": self.tokenizer, "max_seq_length": self.tokenizer.model_max_length},
                                       remove_columns=dataset.column_names)

            # from datasets import Dataset
            # import pandas as pd
            # dataset = pd.read_json(dataPath)
            # dataset = Dataset.from_pandas(dataset)
            # self.dataset = dataset.map(formatting_prompts_func_openmind, fn_kwargs={"tokenizer": self.tokenizer,
            #                             "max_seq_length": self.tokenizer.model_max_length},
            #                             remove_columns=dataset.column_names)

    def mergeQloraModel(self, base_model_path, lora_model_path, merge_out_path):

        base_model, tokenizer = self.loadModel(base_model_path)
        lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
        merged_model = lora_model.merge_and_unload()

        question = "患者表现为干咳，或咯少量粘痰，有时痰中带血，胸部隐痛，午后手足心热，皮肤干灼，或有盗汗，舌质红苔薄，脉细数。请问该患者的中医辨证是什么？"
        print("-------------------------------base model infer result-------------------------------")
        self.deepSeekInfer(base_model, tokenizer, question)
        print("-------------------------------merge model infer result------------------------------")
        self.deepSeekInfer(merged_model, tokenizer, question)

        # 保存合并后的模型
        tokenizer.save_pretrained(merge_out_path)
        merged_model.save_pretrained(merge_out_path, safe_serialization=False)

    def fineTune(self):
        # 加载模型、
        self.model, self.tokenizer = self.loadModel(self.args.base_model_path)
        # 构建微调数据
        self.dataLoad(self.args.data_path)
        # 微调
        from swanlab.integration.transformers import SwanLabCallback
        swanlab_config = {
            "dataset": self.args.data_path,
            "peft": "lora"
        }
        swanlab_callback = SwanLabCallback(
            project="deepseek-finetune-test",
            experiment_name="DeepSeek-R1-Distill-Qwen-7B",
            description="deepSeek-r1蒸馏模型微调",
            workspace=None,
            config=swanlab_config
        )

        if self.args.fine_tune_type == "unsloth":
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from unsloth import is_bfloat16_supported

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = 16,
                lora_alpha=16,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj",],
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 917,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
            trainer = SFTTrainer(
                model = self.model,
                tokenizer = self.tokenizer,
                train_dataset = self.dataset,
                dataset_text_field = "text",
                max_seq_length = self.max_seq_length,
                dataset_num_proc = self.args.num_train_epochs,
                packing = False, # Can make training 5x faster for short sequences.
                callbacks=[swanlab_callback],
                args = TrainingArguments(
                    per_device_train_batch_size = self.args.batch,
                    gradient_accumulation_steps = self.args.accumulation_steps,
                    warmup_steps = 5,
                    num_train_epochs = self.args.num_train_epochs,
                    learning_rate = 2e-4,
                    fp16 = not is_bfloat16_supported(),
                    bf16 = is_bfloat16_supported(),
                    logging_steps = self.args.logging_steps,
                    optim = "adamw_8bit",
                    weight_decay = 0.01,
                    lr_scheduler_type = "linear",
                    seed = 917,
                    output_dir = self.args.out_path,
                    save_steps=self.args.save_steps,
                    max_steps=self.args.max_steps,
                    report_to = "none", # Use this for WandB etc
                )
            )
            # 训练
            trainer.train()

        else:
            if self.args.fine_tune_type == "openMind":
                from openmind import TrainingArguments
                from openmind import Trainer
            else:
                from transformers import TrainingArguments
                from transformers import Trainer

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                inference_mode=False
            )

            # 配置训练参数
            train_args = TrainingArguments(
                output_dir=self.args.out_path,
                per_device_train_batch_size=self.args.batch,
                gradient_accumulation_steps=self.args.accumulation_steps,
                logging_steps=self.args.logging_steps,
                num_train_epochs=self.args.num_train_epochs,
                save_steps=self.args.save_steps,
                max_steps = self.args.max_steps,
                learning_rate=2e-4,
                save_on_each_node=True,
                gradient_checkpointing=True,
                report_to=None,
                seed=917,
                optim="adamw_torch",
                fp16=True,
                bf16=False,
                remove_unused_columns=False,
            )

            # 用于确保模型的词嵌入层参与训练
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

            # 配置训练器
            if self.args.fine_tune_type == "openMind":
                data_collator = DataCollatorForSeq2SeqCustom(tokenizer=self.tokenizer, padding=True, return_tensors="pt")
                trainer = Trainer(
                    model=self.model,
                    args=train_args,
                    train_dataset=self.dataset,
                    data_collator=data_collator,
                    callbacks=[swanlab_callback],
                )
            else:
                trainer = Trainer(
                    model=self.model,
                    args=train_args,
                    train_dataset=self.dataset,
                    callbacks=[swanlab_callback],
                )
            trainer.train()

        self.mergeQloraModel(self.args.base_model_path, self.args.lora_model_path, self.args.merge_model_path)
def config_parameters():
    parser = argparse.ArgumentParser(description="Lora fine-tuning for DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--base_model_path", type=str, default="./baseModels/DeepSeek-R1-Distill-Qwen-7B", help="The path of the basic model file!")
    parser.add_argument("--out_path", type=str, default="./loraModels/DeepSeek-R1-Distill-Qwen-7B", help="The path of the lora output file!")
    parser.add_argument("--lora_model_path", type=str, default="./loraModels/DeepSeek-R1-Distill-Qwen-7B/checkpoint-200", help="The path of the lora checkpoint file!")
    parser.add_argument("--merge_model_path", type=str, default="./mergeModels/DeepSeek-R1-Distill-Qwen-7B", help="The path of merge model output file!")
    parser.add_argument("--data_path", type=str, default="./trainData/medical_o1_sft_Chinese.json", help="The path of train data file!")
    parser.add_argument("--data_type", type=str, default="json", help="The type of train data file!")

    parser.add_argument("--fine_tune_type", type=str, default="unsloth", help="Fine tune method!") # transformers/openMind/unsloth
    parser.add_argument("--batch", type=int, default=2, help="Fine tune batch size!")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Fine tune gradient accumulation steps!")
    parser.add_argument("--r", type=int, default=8, help="Fine tune the lora_rank parameter!")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Fine tune the lora_alpha parameter!")

    parser.add_argument("--num_train_epochs", type=int, default=2, help="Fine tune epochs frequency!")
    parser.add_argument("--save_steps", type=int, default=100, help="Interval saving steps!")
    parser.add_argument("--logging_steps", type=int, default=100, help="Interval printing output steps!")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps!")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = config_parameters()
    deepSeekQLoraer = DeepSeekQLora(args)
    deepSeekQLoraer.fineTune()

    # model, tokenizer = deepSeekQLoraer.loadModel(args.base_model_path)
    # question = "患者表现为干咳，或咯少量粘痰，有时痰中带血，胸部隐痛，午后手足心热，皮肤干灼，或有盗汗，舌质红苔薄，脉细数。请问该患者的中医辨证是什么？"
    # deepSeekQLoraer.deepSeekInfer(model, tokenizer, question)
    # deepSeekQLoraer.mergeQloraModel(args.base_model_path, args.lora_model_path, args.merge_model_path)