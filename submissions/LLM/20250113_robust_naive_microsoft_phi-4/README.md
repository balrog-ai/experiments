# BALROG DeepSeek R1 Distill Qwen 32B CoT
We use naive zero-shot agents as the baseline for all the environments in BALROG.

Specifically:

1. The agent is first given the rules of the games and available actions
2. The agent is then shown the history of previous 16 observations-actions in a chat format, 
3. Finally, the agent is asked to only output a single action, with not sophisticated reasoning mechanism. 

To replicate this robust_naive baselines, use `agent.type=robust_naive`:

```
python eval.py \
  agent.type=robust_naive \
  agent.max_image_history=0 \
  eval.num_workers=32 \
  client.client_name=claude \
  client.model_id=claude-3-5-haiku-20241022
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)