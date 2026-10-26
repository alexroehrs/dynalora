# Statistics and LLM-judge results

- `ci_bootstrap.csv`: for every (config, model, method, lambda/gamma/temperature) grid point and metric, the mean and the 95% interval of a two-stage cluster bootstrap (prompts, then generations within each prompt; 1,000 replicates). Reference texts are resampled as independent texts. Produced by `scripts/bootstrap_ci.py` from the per-generation metric files in `../gpt2/` and `../qwen/`.
- `paired_tests.csv`: paired cluster-bootstrap differences between two grid points of the same adapter (same prompts and seeds), with 95% intervals and a two-sided bootstrap p-value.
- `judge_cells_ci.csv`: LLM-judge ratings (1 to 5) per cell of the stratified sample (50 texts per cell), for each judge (`Qwen` = Qwen2.5-7B-Instruct, `OLMo` = OLMo-2-1124-7B-Instruct): mean fluency `F` and genre fit `D` with the half-width of the 95% prompt-clustered bootstrap interval (`Fhw`, `Dhw`). Cells: `dyn<lambda>` DynaLoRA, `w<lambda>` weight-space scaling, `cold` plain adapter at T = 0.5, `cfg` classifier-free guidance at gamma = 1.5, `ref` held-out reference texts. Produced by `scripts/llm_judge.py`.

Inter-judge agreement: fluency, Spearman 0.51 over items and Pearson 0.56 over cell means; genre fit, 0.26 and 0.00.
