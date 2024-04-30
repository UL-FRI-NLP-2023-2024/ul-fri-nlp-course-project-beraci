## 1. Literature Review
**Goal:** Understand the current state of question generation models, especially Doc2Query, and cross-lingual NLP techniques.

**Steps:**
- Research papers on Doc2Query and T5 models, especially how they have been applied in question generation tasks.
- Study cross-lingual NLP methods and their challenges, particularly focusing on multilingual models like mT5.
- Explore existing implementations and case studies, possibly on platforms like Hugging Face or GitHub.

## 2. Dataset Selection and Preparation
**Goal:** Identify and prepare a suitable Slovenian dataset for training and evaluating the model.

**Steps:**
- **Option 1:** Find an existing Slovenian Question Answering (QA) dataset. If one isn’t available in Slovenian, consider translating a subset of an existing dataset like SQuAD using a reliable translation service.
- **Option 2:** Construct a dataset using Slovenian news articles. This involves:
  - Extracting text from news sources.
  - Utilizing existing NLP tools to generate potential questions.
  - Having native speakers validate and correct these questions and their answers.
- Prepare the dataset for model training, which includes tokenization, splitting into train/validation/test sets, and formatting according to T5’s input requirements.

## 3. Model Fine-Tuning
**Goal:** Adapt the T5 model to generate questions in Slovenian, utilizing High-Performance Computing (HPC) resources.

**Steps:**
- Set up the T5 model with the initial Doc2Query weights, if available. If not, start with a pretrained multilingual T5 model.
- Configure your training environment on an HPC to handle the computational requirements.
- Fine-tune the model on your Slovenian dataset, monitoring for convergence and performance issues.
- Save and document different versions of the model during the training process for later analysis.

## 4. Quality Assessment
**Goal:** Evaluate the quality of the generated questions in terms of relevance, coherence, and linguistic correctness.

**Steps:**
- Develop a set of criteria for what makes a "good" question in this context. These might include clarity, relevance to the document, grammatical correctness, and engagement.
- Manually review and annotate a set of 300 examples, as required. Try to ensure multiple reviewers assess each example to reduce bias.
- Use statistical and NLP tools to analyze the linguistic features and correctness of the questions.
- Consider developing an automated evaluation script that can help in scaling the review process.

## 5. Final Report
**Goal:** Summarize your findings, methodologies, and insights gained from the project.

**Steps:**
- Compile the results from the fine-tuning and evaluation phases.
- Discuss the performance of the pre-trained model versus the fine-tuned model.
- Highlight any challenges faced, particularly those specific to the Slovenian language or the cross-lingual aspects of the project.
- Conclude with potential improvements and future research directions.

## 6. References
**Goal:** Properly cite all sources and tools used throughout your project to ensure academic integrity and provide readers with resources for further study.

**Steps:**
- Keep detailed records of all papers, articles, and resources consulted.
- Use a consistent format for citations in your report, as specified by your academic or project guidelines.
