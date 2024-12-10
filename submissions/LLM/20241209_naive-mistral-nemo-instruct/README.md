# BALROG Naive Baseline
We use naive zero-shot agents as the baseline for all the environments in BALROG.

Specifically:

1. The agent is first given the rules of the games and available actions
2. The agent is then shown the history of previous 16 observations-actions in a chat format, 
3. Finally, the agent is asked to only output a single action, with no sophisticated reasoning mechanism. 

To replicate the naive baselines, use `agent.type=naive`:

```
vllm serve mistralai/Mistral-Nemo-Instruct-2407 --port 8080

python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=32 \
  client.client_name=vllm \
  client.model_id=mistralai/Mistral-Nemo-Instruct-2407 \
  client.base_url=http://0.0.0.0:8080/v1
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/assets/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)

