### https://huggingface.co/bkoloski/slv_doc2query?library=true

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "bkoloski/slv_doc2query"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

### za spremenit da bere iz datoteke squad_v2 v mapi data
text = "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."


input_ids = tokenizer.encode(text, max_length=384, truncation=True, return_tensors='pt')
outputs = model.generate(
    input_ids=input_ids,
    max_length=64,
    do_sample=True,
    top_p=0.95,
    num_return_sequences=5)

print("Text:")
print(text)

print("\nGenerated Queries:")
for i in range(len(outputs)):
    query = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(f'{i + 1}: {query}')
