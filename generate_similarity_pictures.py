import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the provided JSON files
with open('result1.json', 'r', encoding='utf-8') as f1, open('result2.json', 'r', encoding='utf-8') as f2, open('result3.json', 'r', encoding='utf-8') as f3:
    data1 = json.load(f1)
    data2 = json.load(f2)
    data3 = json.load(f3)

# Combine the data
data = data1 + data2 + data3

# Extract necessary details for analysis
right_questions = [entry['right_question'] for entry in data]
closest_questions = [entry['closest_question'] for entry in data]
similarity_scores = [entry['similarity_scores'] for entry in data]
scores = [entry['score'] for entry in data]

# Flatten similarity scores for detailed analysis
flat_similarity_scores = [score for sublist in similarity_scores for score in sublist]

# Create DataFrame for detailed similarity scores
df_similarity = pd.DataFrame(flat_similarity_scores, columns=['Similarity Score'])

# Create summary DataFrame for scores
df_scores = pd.DataFrame({
    'Right Question': right_questions,
    'Closest Question': closest_questions,
    'Score': scores
})

# Plot distribution of similarity scores
plt.figure(figsize=(10, 6))
sns.histplot(flat_similarity_scores, bins=20, kde=True)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.savefig('similarity_scores_distribution.png')
plt.show()

# Plot summary statistics of scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_scores, x='Score')
plt.title('Boxplot of Generated Question Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.savefig('scores_boxplot.png')
plt.show()

# Calculate average similarity scores for each right question
average_similarity_scores = [sum(scores)/len(scores) for scores in similarity_scores]

# Plot average similarity scores
plt.figure(figsize=(14, 8))
sns.barplot(x=range(len(average_similarity_scores)), y=average_similarity_scores)
plt.title('Average Similarity Scores per Right Question')
plt.xlabel('Question Index')
plt.ylabel('Average Similarity Score')
plt.savefig('average_similarity_scores.png')
plt.show()
