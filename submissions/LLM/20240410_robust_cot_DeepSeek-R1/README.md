# BALROG DeepSeek R1 Chain of Thought
To replicate DeepSeek R1 with the robust chain of thought agent template using NVIDIA NIM


export OPENAI_API_KEY=YOUR-NVIDIA-NIM-API-KEY

```
python3 -m eval \
  agent.type=robust_cot \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=4 \
  client.client_name=nvidia \
  client.model_id=deepseek-ai/DeepSeek-R1
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)