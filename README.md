# Investigating Attention and Decoding Strategies for Neural Dialogue Summarization

This course project studies recurrent neural approaches to abstractive dialogue
summarization on the [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum).
It compares a vanilla LSTM encoder--decoder with a Bahdanau-attention model and
evaluates greedy decoding against beam search.

## Compared configurations

- Vanilla one-layer unidirectional LSTM Seq2Seq
- LSTM Seq2Seq with masked Bahdanau/additive encoder--decoder attention
- Greedy decoding, beam search with width 3, and beam search with width 5

## Final test results

ROUGE values are mean example-level F-measures. Generated length is in tokens.

| Model | Decoding | R1 | R2 | RL | Gen. mean length |
|---|---|---:|---:|---:|---:|
| Vanilla | Greedy | .1233 | .0131 | .1075 | 13.60 |
| Vanilla | Beam-3 | .1235 | .0163 | .1022 | 12.42 |
| Vanilla | Beam-5 | .1062 | .0141 | .0889 | 11.74 |
| Attention | Greedy | .2838 | .0794 | .2399 | 15.32 |
| Attention | Beam-3 | .2771 | .0835 | .2379 | 12.78 |
| Attention | Beam-5 | .2698 | .0806 | .2339 | 12.41 |

Detailed final metrics are available in
[`results/final_test_evaluation/`](results/final_test_evaluation/).

## Key findings

- Bahdanau attention substantially improved ROUGE over the vanilla Seq2Seq baseline.
- Beam search did not provide an overall improvement over greedy decoding and generated shorter summaries.
- Attention outperformed the vanilla model across short, medium, and long dialogues, although its advantage decreased as dialogue length increased.

## Repository structure

```text
src/         Model, preprocessing, training, decoding, and evaluation modules
scripts/     Experiment, evaluation, and plotting scripts
notebooks/   Data exploration and validation-analysis notebook
results/     Saved validation and final test metrics
report/      LaTeX report, tables, and figures
```

## Setup
```text
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt 
```
Open notebooks/analysis.ipynb for the exploration and validation workflow.