# Metodologia consolidada

## Fontes canônicas

A análise final deve usar os artefatos em `output/canonical/`, gerados por:

```powershell
$env:PYTHONPATH = "fases/01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.consolidacao_resultados `
  --evidence-root fases\01_validacao_conceitual\evidencias\benchmarks `
  --output output\canonical
```

Fontes aceitas:

- `robustez_sintetica_v3_2026-09-13`;
- `hardware_nativo_v3_2026-09-14`;
- `hardware_stress_complementar_v3_2026-09-17`;
- `modelos_opt_v1_2026-09-15`;
- `modelos_opt_2_7b_2026-09-15`;
- `modelos_opt_6_7b_2026-09-16`;
- `modelos_opt_complementar_lacunas_2026-09-17`;
- `modelos_opt_6_7b_complementar_lacunas_2026-09-17`.

Fontes excluídas das conclusões:

- smokes;
- diagnósticos CPU antigos;
- benchmark GPU v1 quando há evidência v3 para a mesma pergunta;
- duplicatas de qualidade substituídas por runs complementares mais recentes.

## Regra de precedência

Cada linha recebe `canonical_source` e `canonical_source_priority`. Quando duas
fontes descrevem a mesma chave experimental, a fonte de maior prioridade vence;
empates preservam a última fonte lida. As linhas descartadas ficam em
`discarded_*_duplicates.csv`, não são apagadas.

## Níveis de evidência

- `A_numeric_change`: altera pesos ou saídas numericamente, mas não garante
  armazenamento compacto nem kernel físico. Inclui INT8 fake e pruning denso
  percentual.
- `B_physical_compression`: reduz armazenamento físico, sem promessa automática
  de latência. Inclui INT8 weight-only.
- `C_physical_acceleration_candidate`: usa caminho que pode acelerar execução,
  mas só sustenta conclusão física quando `physical_kernel_status=confirmed`.
  Inclui INT8 dynamic e pruning 2:4.

## Nomenclatura de latência

O campo legado `operation=model_ttft` foi preservado para compatibilidade com os
CSVs anteriores. A coluna nova `measurement_label` descreve a métrica como
`prefill_to_first_logit`, porque o protocolo atual mede o forward do prompt até
o primeiro logit, não TTFT completo de sistema com tokenização, scheduling e
streaming.

## Qualidade

O critério oficial permanece:

- quantização aceitável: `Delta PPL <= 2%`;
- pruning aceitável: `Delta PPL <= 5%`.

Os campos `delta_ppl_le_2pct`, `delta_ppl_le_5pct` e `delta_ppl_le_10pct` são
exploratórios. Eles ajudam a discutir fronteiras de Pareto, mas não mudam o
critério oficial já registrado no projeto.

## Seletor fino OPT

Os cenários de OPT agora aceitam seleção por camada e componente:

- `layer_start` e `layer_end`: intervalo fechado de camadas;
- `layers`: lista explícita de camadas;
- `components`: subconjunto de `attention`, `mlp`, `q_proj`, `k_proj`,
  `v_proj`, `out_proj`, `fc1`, `fc2`.

Isso permite testar sensibilidade por região do Transformer e preparar
configurações híbridas sem reescrever o runner principal.

## Configurações novas preparadas

- `hardware_crossover_v3.json`: microbenchmark físico mínimo para observar
  crossover por sequência, batch e largura.
- `modelos_opt_sensibilidade_350m.json`: triagem barata de sensibilidade por
  componente/região em OPT-350M.
- `modelos_opt_sensibilidade_1_3b.json`: confirmação intermediária em OPT-1.3B.

Essas configurações estão prontas para execução, mas não devem ser confundidas
com evidência já coletada enquanto seus manifestos de resultado não existirem.
