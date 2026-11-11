# Zenodo: метаданные и описание

DOI: 10.5281/zenodo.24143896

## Поля формы

| Поле | Значение |
| --- | --- |
| Resource type | Publication / Preprint |
| Title | Zero-cost curvature proxies for adaptive stepping in flow matching: a calibrated error predictor, a static grid, and no measurable headroom for per-sample control |
| Publication date | дата публикации |
| Version | 1.0 |
| Language | English |
| Authors | Savelev, Iakov (GitHub: YakovSava). Второго автора добавьте сами: Yuzvynsky, Yehor, если он указывается как соавтор |
| License | Creative Commons Attribution 4.0 (текст); код в репозитории под MIT |
| Keywords | flow matching; ODE sampler; adaptive step size; step schedule; truncation error; curvature; classifier-free guidance; bfloat16; ComfyUI; diffusion models; negative result |
| Related identifiers | "Continues": 10.5281/zenodo.23529356 (DOI, adaptive-step-samplers: бесплатная оценка ошибки усечения). "Is supplemented by": https://github.com/YakovSava/zero-cost-proxy-research (URL). "References": arXiv:2609.01746 (CAT-Flow, ближайшая работа). "References": 10.5281/zenodo.23337579, 10.5281/zenodo.23939696 (по желанию) |
| References | Вставьте список из `publication/references_apa.md`, по одной ссылке на строку |

Перед публикацией:

1. Выполните коммит и `git push` в `origin` (https://github.com/YakovSava/zero-cost-proxy-research). Сейчас всё лежит в рабочей папке незакоммиченным, поэтому ссылки на GitHub в текстах заработают только после этого. Папка называется `flow_zero_cost_proxy`, репозиторий `zero-cost-proxy-research`: имя в README и текстах везде репозиторное.
2. Положите в запись README.md и `figures/` (минимум `grid_face_s42.jpg`, `grid_fur_s42.jpg`, `error_vs_first_step.png`).
3. Решите, указывать ли помощь ИИ. Код, анализ и черновики текстов подготовлены с ассистентом Claude (Anthropic) по вашим указаниям. Правила Zenodo, журналов и площадок (Medium, Habr) всё чаще требуют раскрытия такого использования. Готовая формулировка есть в конце описания ниже (отмечена как необязательная).
4. Проверьте уровни B-D в `references_apa.md`.

## Description (HTML для вставки в режиме Source)

```html
<p><b>Summary.</b> A multistep flow-matching sampler already holds two consecutive velocity evaluations, so the angle between them is a free estimate of how fast the trajectory bends. We tested this "zero-cost curvature proxy" on the Anima flow model (bf16, CFG 5, 576x768, 12 units = 4 prompts x 3 seeds) against the true local error computed from a converged trajectory, and then asked whether using it to control the step size helps at 8-24 network evaluations. The proxy is a good predictor of the error. As a way to choose steps, what matters is the shape of the step grid, not the per-sample adaptation.</p>

<p><b>Main results.</b></p>
<ul>
<li>The cosine between consecutive velocities measures rotation but not change of speed (exact identity: |dv|^2 = (ds)^2 + 4 s s' sin^2(theta/2)). On Anima, speed change carries 0.3-0.4% of the acceleration energy in the bulk of the trajectory and about 8-9% (up to 23%) in the first and last 5% of the time.</li>
<li>The estimate E = (h/2) |v| theta predicts the true next-step Euler error with a median ratio of 0.80-0.84 and an across-unit log-correlation of 0.91-0.92, for steps 0.01-0.12 of the unit time interval.</li>
<li>Spacing the evaluations by equal turning angle (grid density proportional to the leave-one-out mean angle rate |v'|/|v|) lowers the final-latent error against a converged solution by 30-43% compared with the default ComfyUI "normal" schedule at equal NFE (Euler, 8-24 evaluations, 12 units; paired bootstrap intervals exclude zero in every cell). Weighting that density by a measured error-propagation profile gives 41-50%. The asymptotic Euler optimum (exponent 1/2) is significantly better than "normal" only at 24 evaluations.</li>
<li>Per-sample adaptation added nothing measurable. A step grid produced by the controller on another sample gives the same error as the sample's own (log-ratio -0.01 to -0.02, 95% interval about +-0.07), and the controller with a small initial step equals the static grid it implies (log-ratio -0.02, interval -0.09 to +0.04). The outcome of the controller is set mostly by its blind first step (mean error 0.80, 0.51 and 0.26 for initial steps 0.06, 0.02 and 0.01).</li>
<li>In bf16 with CFG 5 the velocity carries evaluation noise of about 1.7 degrees (3.4 degrees in the bulk of a 96-step trajectory) that does not shrink with the step; it comes from rounding (a plateau for input perturbations of 1e-5 to 1e-3), and CFG multiplies it by about 4. A raw cosine controller with a target angle below this floor collapses to its minimum step. Shifting the sigmas of a sampler by a relative 1e-7 changes the final latent by 3-10%.</li>
<li>The result depends on the metric. With LPIPS to the converged image, the best static grid wins on three prompt classes (39-48% lower, geometric mean over 8, 12 and 16 evaluations) and loses on the texture prompt (about 16% higher) where latent L2 still improves. Adams-Bashforth 2 does not benefit from these grids: their rapid step growth at the start makes it worse than Euler.</li>
</ul>

<p><b>What is not new.</b> Training-free, zero-extra-evaluation, curvature-adaptive step sizes for flow matching have been proposed before (CAT-Flow, arXiv:2609.01746, which we found only after the experiments and did not compare against), and static schedules fitted to a model or to reference trajectories exist (Align Your Steps, Optimal Stepsize Distillation, Bespoke Solvers). What the record adds is the decomposition of what the cosine measures, the calibration of the estimator against the true error on a real model, the noise-floor measurement, and the test of per-sample adaptation against borrowed and oracle grids.</p>

<p><b>Setup.</b> Anima (CircleStone Labs / Comfy Org, bf16), 576x768, CFG 5, ComfyUI "normal" scheduler as the baseline, Euler and Adams-Bashforth 2. The reference is Adams-Bashforth 2 with 192 uniform points (about 5% error of its own). A synthetic control with the exact velocity field of a Gaussian mixture (D = 64 and 256) is included. 30 unit tests.</p>

<p><b>Limitations.</b> One model, one resolution, one CFG scale, 12 units; effects below about 5-8% are not detectable. The main metric is relative L2 distance of the final latent to a converged solution; LPIPS was checked for one grid only. The hypotheses were not pre-registered in a file before the measurements, and the first-step finding is post hoc (the control experiments were designed after it). The propagation profile used by the best grid was measured on two of the 12 evaluation units, so the rows that use it are slightly optimistic; the plain angle-rate grids are leave-one-out. The study covers deterministic Euler and Adams-Bashforth 2 only.</p>

<p><b>Software.</b> Code, tests, tables (<code>results/</code>), figures and ready-made sigma lists for Anima are in the repository: <a href="https://github.com/YakovSava/zero-cost-proxy-research">https://github.com/YakovSava/zero-cost-proxy-research</a>. The full record, with the hypothesis table and all intervals, is the README.</p>

<p><i>AI assistance (optional, edit to taste).</i> The code, the analysis scripts and the first drafts of the texts were produced with an AI coding assistant (Claude, Anthropic) under the author's direction. The author checked the results and is responsible for the content.</p>
```
