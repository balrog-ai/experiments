# BALROG Gemini-2.5-Pro-Exp
To replicate Gemini-2.5-Pro-Exp with the naive agent template, use:

```
export GEMINI_API_KEY=<KEY>

python3 -m eval \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=16 \
  client.client_name=gemini \
  client.model_id=gemini-2.5-pro-preview-03-25
```

Bare in mind that Gemini 2.5 Pro is using some reasoning internally

For more information, visit:

- [Evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md)
- [Paper](https://arxiv.org/abs/2411.13543)
- [BALROG](https://balrogai.com)