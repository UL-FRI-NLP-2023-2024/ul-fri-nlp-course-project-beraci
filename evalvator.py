from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# file procesing
def process_files(right_questions_file, sample_questions_file, result_file):
    # 100 selected questions
    right_questions = []
    with open(right_questions_file, encoding="utf8") as f:
        data = json.load(f)
        query_data = data["data"]
        for record in query_data:
            right_questions.append(record["question"])

    # 5 sample questions per each selected question generated by the model
    with open(sample_questions_file, 'r', encoding='utf-8') as file:
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

        # podobnost
        similarities = cosine_similarity(right_question_embedding, candidate_embeddings).flatten()

        closest_index = np.argmax(similarities)
        closest_question = candidate_questions[closest_index]

        results.append({
            'right_question': right_question,
            'closest_question': closest_question,
            'similarity_scores': similarities.tolist(),
            'sample_questions': questions[i*5:(i+1)*5]
        })

    # save to json
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

#iterate over files
for i in range(1, 4):
    right_questions_file = f'data/questions{i}.json'
    sample_questions_file = f'output/generated_queries/questions{i}_generated.txt'
    result_file = f'result{i}.json'
    
    process_files(right_questions_file, sample_questions_file, result_file)
