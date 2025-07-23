# BALROG Grok-4
To replicate grok-4 with the naive agent template, use:

```
export OPENAI_API_KEY=YOUR-XAI-API-KEY

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=8 \
  client.client_name=xai \
  client.base_url="https://api.x.ai/v1" \
  client.model_id=grok-4-latest
```

Bare in mind that while this is using the naive agent, Grok4 could be reasoning before replying.

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)