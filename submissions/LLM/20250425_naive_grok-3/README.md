# BALROG Grok-3-beta
To replicate grok-3-beta with the naive agent template, use:

```
export OPENAI_API_KEY=YOUR-XAI-API-KEY

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=xai \
  client.base_url="https://api.x.ai/v1" \
  client.model_id=grok-3-beta
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)