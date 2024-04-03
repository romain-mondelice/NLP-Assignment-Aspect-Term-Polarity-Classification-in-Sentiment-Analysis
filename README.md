# NLP-Assignment-Aspect-Term-Polarity-Classification-in-Sentiment-Analysis

## Contributors
- Romain MONDELICE
- Yanis CHAMSON

## Classifier Description
- Type of classification model:
- Input representation:
- Feature representation:
- Resources used:

## Experiments
We conducted four experiments to evaluate the performance of our aspect-based sentiment analysis classifier. The experiments involved fine-tuning BERT and RoBERTa models with different preprocessing techniques for the input text.

### Experiment 1: BERT with Preprocessing Technique 1
- Model: BERT
- Preprocessing: `f"{aspect_category} {target_term} [SEP] {sentence}"`
- Training dataset accuracy:
- Development dataset accuracy:

### Experiment 2: BERT with Preprocessing Technique 2
- Model: BERT
- Preprocessing: `f"What do you think about the {target_term}{aspect_category} ? [SEP] {sentence}"`
- Training dataset accuracy:
- Development dataset accuracy:

### Experiment 3: RoBERTa with Preprocessing Technique 1
- Model: RoBERTa
- Preprocessing: `f"{aspect_category} {target_term} [SEP] {sentence}"`
- Training dataset accuracy:
- Development dataset accuracy:

### Experiment 4: RoBERTa with Preprocessing Technique 2
- Model: RoBERTa
- Preprocessing: `f"What do you think about the {target_term}{aspect_category} ? [SEP] {sentence}"`
- Training dataset accuracy:
- Development dataset accuracy:

## Benchmark Table
| Model    | Preprocessing Technique | Training Accuracy | Development Accuracy |
|----------|-------------------------|-------------------|----------------------|
| BERT     | Technique 1             |                   |                      |
| BERT     | Technique 2             |                   |                      |
| RoBERTa  | Technique 1             |                   |                      |
| RoBERTa  | Technique 2             |                   |                      |

## Conclusion
Provide a brief conclusion summarizing the results and any insights gained from the experiments.