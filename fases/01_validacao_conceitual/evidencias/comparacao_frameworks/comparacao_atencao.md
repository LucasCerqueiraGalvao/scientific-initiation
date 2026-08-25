# Comparação numérica da scaled dot-product attention

> Esta evidência valida correção numérica. Ela não compara desempenho e não sustenta alegações de hardware.

- Referência: `numpy_manual`
- Seed: `2026`
- Dtype e dispositivo: `float32` em `cpu`
- Tolerâncias: `atol=1e-06` e `rtol=1e-05`
- Backend Keras: `tensorflow`

| Caso | Candidato | Máscara | Erro máximo | Erro médio | MSE | Similaridade de cosseno | Passou |
| --- | --- | --- | ---: | ---: | ---: | ---: | :---: |
| tiny_manual_unmasked | pytorch_manual | none | 2.38418579e-07 | 5.96046448e-08 | 1.42108547e-14 | 1.00000000e+00 | sim |
| tiny_manual_unmasked | pytorch_sdpa | none | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | sim |
| tiny_manual_unmasked | keras_tensorflow_sdpa | none | 2.38418579e-07 | 5.96046448e-08 | 1.42108547e-14 | 1.00000000e+00 | sim |
| seeded_multihead_unmasked | pytorch_manual | none | 1.19209290e-07 | 2.45636329e-08 | 1.77382703e-15 | 1.00000000e+00 | sim |
| seeded_multihead_unmasked | pytorch_sdpa | none | 1.19209290e-07 | 2.64262781e-08 | 1.51824444e-15 | 1.00000000e+00 | sim |
| seeded_multihead_unmasked | keras_tensorflow_sdpa | none | 1.19209290e-07 | 2.33218695e-08 | 1.57491208e-15 | 1.00000000e+00 | sim |
| seeded_multihead_additive_mask | pytorch_manual | additive_bias | 5.96046448e-08 | 6.20881716e-09 | 3.33066907e-16 | 1.00000000e+00 | sim |
| seeded_multihead_additive_mask | pytorch_sdpa | additive_bias | 1.19209290e-07 | 2.23517418e-08 | 1.53580852e-15 | 1.00000000e+00 | sim |
| seeded_multihead_additive_mask | keras_tensorflow_sdpa | additive_bias | 1.19209290e-07 | 2.73187955e-08 | 2.09092003e-15 | 1.00000000e+00 | sim |

## Conclusão

Todos os casos ficaram dentro das tolerâncias definidas.

As versões completas do ambiente estão em `comparacao_atencao.metadata.json` e os valores auditáveis em `comparacao_atencao.csv`.
