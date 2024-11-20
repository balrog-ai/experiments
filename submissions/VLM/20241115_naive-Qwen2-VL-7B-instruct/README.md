# BALROG Naive Baseline
As part of the original BALROG paper, we use naive zero-shot agents as the baseline approach for all the environments in BALROG.

Specifically:

1. The agent is first given the rules of the games and available actions
2. The agent is then shown the history of previous 16 observations-actions in a chat format, 
3. Finally, the agent is asked to only output a single action, with not sophisticated reasoning mechanism. 

To replicate the naive baselines, use `agent.type=naive`, an example with GPT4o-mini is:

```
python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  eval.num_workers=32 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/assets/evaluation.md)
- [Paper]()
- [BALROG](https://balrogai.com)