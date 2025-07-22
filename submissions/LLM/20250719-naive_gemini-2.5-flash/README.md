# BALROG Gemini-2.5-Flash
To replicate Gemini-2.5-Flash with the naive agent template, use:

```
export GEMINI_API_KEY=<KEY>

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=16 \
  client.client_name=gemini \
  client.model_id=gemini-2.5-flash
```

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)