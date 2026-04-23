from peft import LoraConfig, TaskType

def main():
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    print(config)

if __name__ == "__main__":
    main()