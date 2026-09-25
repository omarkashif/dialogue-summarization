Goal: Compare a vanilla LSTM Seq2Seq model against an LSTM Seq2Seq model with attention for dialogue summarization on SAMSum dataset. Evaluate using ROUGE-1, ROUGE-2 and ROUGE-L, plus qualitative analysis of generated summaries.



Tokenizer:
    word-level
    lowercase
    punctuation separated

Vocabulary:
    constructed from TRAINING data only
    dialogues + summaries
    min_freq = 2

Special tokens:
    <PAD>
    <UNK>
    <SOS>
    <EOS>

Max dialogue content:
    300 tokens

Max summary content:
    50 tokens

Resulting maximum lengths:
    dialogue = 302 including SOS/EOS
    summary  = 52 including SOS/EOS