# BALROG Reka-Flash-3 (21B) Chain of Thought
To replicate Reka-Flash-3 (21B) with the robust chain of thought agent template:

```
vllm serve RekaAI/reka-flash-3 --tensor_parallel_size 2 --port 8080

python3 eval \
  agent.type=robust_cot \
  eval.num_workers=32 \
  client.client_name=vllm \
  client.model_id=RekaAI/reka-flash-3 \
  client.base_url=http://0.0.0.0:8080/v1
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)