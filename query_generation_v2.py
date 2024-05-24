from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json


#model_name = '/output/main-SLV-google-mt5-base-2024-05-22_21-36-51'
model_name = "D:/fask/NLP/NLP_Project_Main/output/main-SLV-google-mt5-base-2024-05-22_21-36-51"
#model_name = 'D:/fask/NLP/NLP_Project_Main/output/test-SLV-google-mt5-base-2024-05-03_12-01-23'
#model_name = '/d/hpc/home/rs7709/NLP_Project_Main/output/test-SLV-google-mt5-base-2024-05-22_20-00-03'
#model_name = '/d/hpc/home/rs7709/NLP_Project_Main/output/test-SLV-google-mt5-base-2024-05-22_20-15-28'
#model_name = '/d/hpc/home/rs7709/NLP_Project_Main/output/main-SLV-google-mt5-base-2024-05-22_20-29-32'
#model_name = '/d/hpc/home/rs7709/NLP_Project_Main/output/main-SLV-google-mt5-base-2024-05-22_21-36-51'



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#text = "Otok Sanda leži med Škotsko in Severno Irsko, ob južni konici polotoka Kintyre. Skozi leta je bil zgodovinsko povezan z menihi, svetniki in kralji. Obiskala sta ga škotski kralj Robert Bruce in norveški kralj Hacon, njegova kapela pa je povezana s svetim Kolumbo in svetim Ninianom."

def create_queries(para, f):
    
    f = open("questions3_generated.txt", "a", encoding="utf-8")
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

    
    #print("Paragraph:")
    #print(para)
    f.write("Paragraph: ")
    f.write(para)
    
    #print("\nBeam Outputs:")
    f.write("\n\nBeam Outputs:")
    for i in range(len(beam_outputs)):
        query = tokenizer.decode(beam_outputs[i], skip_special_tokens=True)
        #print(f'{i + 1}: {query}')
        f.write("\n")
        f.write(f'{i + 1}: {query}')

    #print("\nSampling Outputs:")
    f.write("\n\nSampling Outputs:")
    for i in range(len(sampling_outputs)):
        query = tokenizer.decode(sampling_outputs[i], skip_special_tokens=True)
        #print(f'{i + 1}: {query}')
        f.write("\n")
        f.write(f'{i + 1}: {query}')
    #print("-------------------------------------------")
    f.write("\n-------------------------------------------\n")
    f.close()
    

print("Starting:")
#f = open('/d/hpc/home/rs7709/NLP_Project_Main/data/questions1.json', encoding="utf8")
f = open('D:/fask/NLP/NLP_Project_Main/data/questions3.json', encoding="utf8")

data = json.load(f)
query_data = data["data"]
n = 0

for record in query_data:
    para = record["context"]
    create_queries(para, f)
    print("(" + str(n) + ")Done with \"" + para[:25] + "\"...")
    n += 1


    



