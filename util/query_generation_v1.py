from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

#model_name = '/output/test-SLV-google-mt5-base-2024-05-03_12-01-23'
model_name = 'D:/fask/NLP/NLP_Project_Main/output/test-SLV-google-mt5-base-2024-05-03_12-01-23'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Otok Sanda leži med Škotsko in Severno Irsko, ob južni konici polotoka Kintyre. Skozi leta je bil zgodovinsko povezan z menihi, svetniki in kralji. Obiskala sta ga škotski kralj Robert Bruce in norveški kralj Hacon, njegova kapela pa je povezana s svetim Kolumbo in svetim Ninianom."

def create_queries(para):
    input_ids = tokenizer.encode(para, return_tensors='pt')
    with torch.no_grad():
        # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
        sampling_outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            top_k=10, 
            num_return_sequences=5
            )
        
        # Here we use Beam-search. It generates better quality queries, but with less diversity
        beam_outputs = model.generate(
            input_ids=input_ids, 
            max_length=64, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            num_return_sequences=5, 
            early_stopping=True
        )


    print("Paragraph:")
    print(para)
    
    print("\nBeam Outputs:")
    for i in range(len(beam_outputs)):
        query = tokenizer.decode(beam_outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')

    print("\nSampling Outputs:")
    for i in range(len(sampling_outputs)):
        query = tokenizer.decode(sampling_outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')

create_queries(text)
