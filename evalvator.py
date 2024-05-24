from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

right_questions = []
with open('data/questions1.json', encoding="utf8") as f:
    data = json.load(f)
    query_data = data["data"]
    for record in query_data:
        right_questions.append(record["question"])

with open('output/generated_queries/questions1_generated.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    questions = []
    start_extraction = False

    for line in lines:
        if "Sampling Outputs:" in line:
            start_extraction = True
        elif "Beam Outputs:" in line:
            start_extraction = False
        elif start_extraction and (line.startswith('1:') or line.startswith('2:') or line.startswith('3:') or line.startswith('4:') or line.startswith('5:') or '?' in line):
            questions.append(line.strip())


model = SentenceTransformer('all-MiniLM-L6-v2')

results = []

for i, right_question in enumerate(right_questions):
    candidate_questions = questions[i*5:(i+1)*5]
    
    right_question_embedding = model.encode([right_question])
    candidate_embeddings = model.encode(candidate_questions)
    
    # kosinusna podobnost
    similarities = cosine_similarity(right_question_embedding, candidate_embeddings).flatten()
    
    closest_index = np.argmax(similarities)
    closest_question = candidate_questions[closest_index]
    
    results.append({
        'right_question': right_question,
        'closest_question': closest_question,
        'similarity_scores': similarities.tolist(),
        'sample_questions' : questions[i*5:(i+1)*5]
    })

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
