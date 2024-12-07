from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

def generate(question: str) -> str:
    # model_id = "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )

    model_id = "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        local_files_only=True, 
        quantization_config=quantization_config, 
        trust_remote_code=True,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    hf = HuggingFacePipeline(pipeline=pipe)

    # hf = HuggingFacePipeline.from_model_id(
    #     model_id="WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B",
    #     task="text-generation",
    #     pipeline_kwargs={"max_new_tokens": 10},
    # )

    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    output = chain.invoke({"question": question})
    with open("output.txt", "w") as f:
        f.write(str(output))
    return output