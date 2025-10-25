# Approach Overview and Key Learnings

**Kaggle - Jigsaw - Agile Community Rules Classification**

The competition provided a dataset where each row included a Reddit comment, its subreddit, one community rule (with duplicates of positive and negative examples), and the task was to predict whether the comment violated the given rule, with the test set containing new examples and rules but no new subreddits.

### Approaches

There are three main approaches:

- Fine-tune a BERT model for binary classification (there's a BERT training pipeline for binary classification)
- Use LoRA to fine-tune an LLM, give it pos and neg examples (a prompt template), then predict the probability that the next word is "yes" or "no”
- Use tokenizer to generate embeddings directly for classification, without learning

My final submission only used the first two approaches. I only started the competition in the last four days, so I didn't have time to research and implement the third method. Some experiments are conducted with the first two approaches.

For the first approach: I roughly got local CV AUC around 0.9-0.94, then increase from 1 to 3 epochs had slight improvement, after that it would decline slightly or have no improvement. I set parameters based on local settings and then used all of train + test's positive/negative examples and labelled training examples from train.csv , could reach around 0.9.

Improvements:

1. **Subreddit**: I started with adding subreddit in differentiating different examples - if it's the same rule but different subreddit, treat them as different examples. This way there would be many more training examples, but also many duplicates. Later saw that the host when selecting examples didn't care about subreddit, only looked at the rule, so the final version considered that even if different subreddit, as long as the text is the same, all are considered duplicates. So the training set included each example only once, PB improved to 0.906.
2. **Negative examples with URLs**: I set negative examples containing links to have target=0.25, PB improved to 0.909.

Regarding LLM experiments: the public notebook result is if you only use train.csv for training, then fine-tune with Qwen2.5-32b (needs >24GB A6000), one epoch can reach around 0.872 PB. If multiple epochs it will decline slightly. Then I started trying Qwen2.5-7b-gptq-int8 and gemma3. If only training locally then loading for inference, inference time is around 2h, and PB only 0.6-0.7, very unsatisfactory. So I switched to complete test-time training. With one example case reached around 0.880 PB for 2 epochs, around 0.888 for 5 epochs, not much improvement but too much time consumed. Then tried Qwen3-4b, with one example case reached 0.898. Then increased gemma3 examples to 2, trained 2 epochs, and reached 0.906 PB.

### Final Solution

The final submission consisted of two LLMs—one gemma3-1b-it and one qwen3-4b-base—plus one BERT model, which is Deberta-v3-base.

1. **Gemma3-1b-it**: 2 training epochs, 3 pos/neg samples, train batch=1, gradient accumulation=8
2. **qwen3-4b-base**: 2 training epochs, 2 pos/neg samples, train batch=1, gradient accumulation=8
3. **Deberta-v3-base**: 3 training epochs, train batch=24

Both LLMs perform inference once (tried with double inference with different samples but submission didn’t go through), then the results are averaged. The BERT model doesn't use examples—each comment is processed independently.

I used 2x T4 GPUs: the first GPU trains and infers gemma3, then bert (approximately 2 hours total), while the second GPU only handles qwen training and inference.

The predictions are weighted after converting each submission CSV file to ranks. Final blend equals 0.4 times rank of gemma3 inference plus 0.3 times rank of bert plus 0.25 times rank of qwen3 inference.

### Thoughts

1. Considered using public datasets to augment the training data, and using a locally trained model to generate manually filtered pseudo-labels. However, it seems all comments in public datasets that violate the rules have been removed/deleted.
2. Considered training specific models for each individual rule (but later felt that training a single model could learn more about what constitutes a "violation" in general, which might be beneficial for generalizability).
3. For certain rules like "no spoilers" in movie-related subreddits, noticed that some comments with "spoiler" content have special markings in the comments. Not sure how those are treated—whether as violations or not—in the test dataset.



