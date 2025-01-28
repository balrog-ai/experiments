# BALROG DeepSeek R1 Distill Qwen 32B Chain of Thought
To replicate DeepSeek R1 distill Qwen 32B with the robust chain of thought agent template:

```
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --port 8080

python3 -m eval \
  agent.type=robust_cot \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=4 \
  client.client_name=vllm \
  client.model_id=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)