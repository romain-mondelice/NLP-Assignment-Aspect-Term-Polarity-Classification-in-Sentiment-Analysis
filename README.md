# NLP-Assignment-Aspect-Term-Polarity-Classification-in-Sentiment-Analysis

## Contributors

- Romain MNDELICE
- Yanis CHAMSON

## Classifier Description

- For our classification, we based ourselves on the paper "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (Chi Sun et. al.). Indeed, one of the main difficulties of this paper was to be able to incorporate the "aspect categories" and "target term" into our model.
  In the end, we chose two pre-trained models: "bert-base-uncase" and "RoBerta", and chose to use Bert as a question-answer manner in order to deal with aspect categories and target terms.
  Indeed, using the [SEP] token BERT is able to distinguish between different segments of text, which was essential for our method. Specifically, we formulated the aspect category and target term as a question, and the review sentence as the answer.

- In order to run the experiments described below, we used a Nvidia 2080 ti GPU.

### Experiment 1: BERT with Preprocessing Technique 1

- Model: BERT
- Preprocessing: `f"{aspect_category} {target_term} [SEP] {sentence}"`

5 epochs with 3 iterations:

- Development dataset accuracy: [83.78, 83.48, 84.57]

**Total mean Development Acc: 84.04%**

**Exec time: 9781.18 sec (3260 per run)**

### Experiment 2: BERT with Preprocessing Technique 2

- Model: BERT
- Preprocessing: `f"What do you think about the {target_term} {aspect_category} ? [SEP] {sentence}"`

5 epochs with 3 iterations:

- Development dataset accuracy: [85.64, 83.78, 82.18]

**Total mean Development Acc: 83.87%**

**Exec time: 9809.97 sec (3269 per run)**

### Experiment 3: RoBERTa with Preprocessing Technique 1

- Model: RoBERTa
- Preprocessing: `f"{aspect_category} {target_term} [SEP] {sentence}"`

5 epochs with 3 iterations:

- Development dataset accuracy: [85.37, 85.9, 85.64]

**Total mean Development Acc: 85.64%**

**Exec time: 10073.87 sec (3357 per run)**

### Experiment 4: RoBERTa with Preprocessing Technique 2

- Model: RoBERTa
- Preprocessing: `f"What do you think about the {target_term}{aspect_category} ? [SEP] {sentence}"`

5 epochs with 3 iterations:

- Development dataset accuracy: [86.44, 85.64, 89.1]

**Total mean Development Acc: 87.06%**

**Exec time: 9912.71 sec (3304 per run)**

## Benchmark Table

| Model   | Experiment  | Development Accuracy |
| ------- | ----------- | -------------------- |
| BERT    | Technique 1 | 84.04%               |
| BERT    | Technique 2 | 83.87%               |
| RoBERTa | Technique 1 | 85.64%               |
| RoBERTa | Technique 2 | 87.06%               |

## Conclusion

In our experimnts, RoBERTa outperforms BERT in Aspect-Based Sentiment Analysis due to its more extensive training on a larger corpus and its focus on comprehensive masked language modeling, which provides a deeper understanding of linguistic nuances and sentiment. This robust training allows RoBERTa to more accurately interpret complex sentence structures and sentiments related to specific aspects or target terms.
