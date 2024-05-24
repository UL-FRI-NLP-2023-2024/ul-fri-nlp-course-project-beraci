import json

f = open('D:/fask/NLP/NLP_Project_Main/data/squad2-slo-mt-dev-short.json', encoding="utf8")
    
data = json.load(f)

f2 = open('D:/fask/NLP/NLP_Project_Main/data/questions2.json', "a", encoding="utf8")

data_list = []
data_dict = {}
x = 0
query_data = data["data"]
for record in query_data:
    answers = record["answers"]["text"]
    if (x == 100):
        break
    if (len(answers) > 0):
        data_list.append(record)
        x += 1
data_dict["data"] = data_list
json.dump(data_dict, f2)