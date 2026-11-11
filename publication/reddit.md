# Reddit

Правила перед публикацией: прочитайте правила каждого сабреддита (самореклама, обязательные теги, дни для проектов, требования раскрывать помощь ИИ). Картинки прикладывайте файлами из `figures/`: для r/comfyui и r/StableDiffusion главная `grid_face_s42.jpg`, вторая `grid_fur_s42.jpg` (контрпример нужен, чтобы пост не выглядел рекламой). Ссылки на репозиторий и препринт дайте в тексте и продублируйте в первом комментарии, если сабреддит режет посты со ссылками. Перед публикацией создайте репозиторий и сделайте `git push`. Фразы про помощь ИИ в конце каждого поста необязательны.

---

## 1. r/comfyui (и r/StableDiffusion с тем же текстом)

**Title:** Step spacing mattered more than the sampler for Anima at 8-16 steps: a free "equal turning angle" grid gives 30-43% lower latent error than `normal` (Euler, 12 images, with a texture counterexample)

**Body:**

I wanted to know whether the angle between consecutive velocities, which a sampler has for free, is a usable signal for choosing step sizes. I measured it on Anima (bf16, CFG 5, 576x768) against a converged reference. Short version:

- **As an error predictor it is good.** `(h/2) * |v| * angle` predicts the true error of one Euler step with a median ratio of 0.8 and a 0.91-0.92 correlation across images.
- **As a per-image controller it did not help.** A controller that shrinks the step when the angle grows lost to the default `normal` schedule at equal step count, and its result depended mostly on its first step (which is blind). Applying the step grid made for one image to a different image gave the same error as the image's own grid.
- **As a fixed grid it works.** Put the steps proportional to the average angle rate over a set of reference runs (equal turning angle per step) and run plain `euler`: the final-latent error against a converged solution is 30-43% lower than `normal` at 8, 12, 16 and 24 steps (12 images, 4 prompts x 3 seeds, paired bootstrap intervals exclude zero). With an extra weight for how errors propagate to the end, 8 steps give a slightly lower error than `normal` at 24 (0.35 vs 0.38).
- **What the grid looks like:** very small first steps, big steps in the middle, smaller again at the end. For 8 steps the plain equal-angle sigmas are `1, 0.98589, 0.9487, 0.83139, 0.59419, 0.22593, 0.04039, 0.0089, 0` (built from all 12 images, so in-sample; the weighted lists and the lists for 12, 16, 24 are in the repo).
- **Where it fails:** I checked LPIPS only for the weighted grid. On the texture-heavy prompt (foliage) it is about 16% worse than `normal`, even though latent distance improves: the grid aligns composition better and loses fine texture. On the face prompt `normal` draws a different framing at 8, 12 and 16 steps, and the weighted grid gets the composition at 8. I did not evaluate the plain equal-angle grid on LPIPS, so I do not know which side of this it falls on.

You can try it without any node: put the sigma list into a manual-sigmas node and use the stock `euler` sampler. It is valid only for Anima at 576x768 with CFG 5 and I tested only Euler. Adams-Bashforth 2 gets worse on these grids, not better.

What this is not: one model, 12 images, no human preference test, and the main metric is distance in latent space. Hypotheses were not pre-registered; the first-step finding was post hoc. Also, the idea itself is not new: CAT-Flow (arXiv 2609.01746) already proposes zero-extra-evaluation curvature-adaptive steps for flow models, and static schedules fitted to a model exist too (Align Your Steps, Optimal Stepsize Distillation). I found CAT-Flow after the experiments and did not compare with it. What I think is new is the test that per-image adaptation adds nothing on top of a static grid on this model, and the measurements of the bf16 noise floor.

Repo with code, tests, tables and sigma lists: https://github.com/YakovSava/zero-cost-proxy-research
Write-up: https://doi.org/10.5281/zenodo.24143896

*(Code and first drafts were produced with an AI coding assistant, Claude, under my direction; I checked the results.)*

---

## 2. r/MachineLearning

**Title:** [R] A zero-cost curvature proxy predicts Euler step error for flow matching, but per-sample step control adds nothing over a static grid (Anima, 12 units; related to CAT-Flow)

**Body:**

**Question.** A multistep sampler has consecutive velocity evaluations for free. Is the angle between them a usable curvature signal for adaptive stepping, and does using it online beat a fixed grid?

**Setup.** Anima (flow model, bf16), 576x768, CFG 5, 4 prompts x 3 seeds = 12 units. Reference: Adams-Bashforth 2 with 192 uniform points (about 5% own error). Baseline: ComfyUI `normal` schedule (shift 3). Euler and AB2, 8-24 evaluations. Paired log-ratios with 95% bootstrap intervals over units. A synthetic control on the exact velocity field of a Gaussian mixture (D = 64, 256). The hypotheses were not pre-registered in a file; the first-step finding is post hoc.

**Results.**
1. Exact identity `|dv|^2 = (ds)^2 + 4 s s' sin^2(theta/2)`: the cosine sees rotation only. Speed change is 0.3-0.4% of acceleration energy in the bulk, 8-9% (max 23%) in the first and last 5% of time.
2. `E = (h/2)|v| theta` predicts the true next-step Euler error: median ratio 0.80-0.84, across-unit log-correlation 0.91-0.92 (steps 0.01-0.12).
3. Evaluation noise in bf16 + CFG 5: about 1.7 degrees between consecutive velocities, not decreasing with h. A plateau for input perturbations 1e-5..1e-3, so rounding, not the flow. CFG multiplies it ~4x. A raw-angle controller with target below this floor collapses to its minimum step. The norm of a derivative estimated from noisy values in D = 110592 is biased upward by the noise power.
4. A cosine controller (`h <- h * (theta*/theta)^gamma`) loses to `normal` at equal NFE with a blind first step h0 = 0.06 (mean error 0.80), and is mostly set by h0 (0.51 at 0.02, 0.26 at 0.01), nearly independent of theta* and NFE.
5. A static grid with density proportional to the leave-one-out mean angle rate (exponent 1, equal angle per step) vs `normal` at equal N: log-ratio -0.39, -0.56, -0.57, -0.36 for N = 8, 12, 16, 24 (30-43% lower), intervals exclude 0. The asymptotic exponent 1/2 is significant only at N = 24. Weighting by an error-propagation profile (two units, measured by large state kicks; small kicks are floor-dominated) gives 41-50%.
6. No headroom for per-sample adaptation: a grid from another sample's controller run vs the sample's own: log-ratio -0.01..-0.02 (interval about +-0.07); controller (h0 = 0.01) vs the static grid it implies: -0.02 [-0.09, +0.04]; per-sample oracle vs static: within noise. In the exact Gaussian-mixture toy the oracle gain is below 0.6% for Euler.
7. AB2 is worse than Euler on these grids (correlation 0.74-0.96 between its error ratio and the step ratio h1/h0).
8. LPIPS (AlexNet) to the converged image for the weighted grid: log-ratio -0.39, -0.36, -0.43 at N = 8, 12, 16 (wins 11/12, 9/12, 9/12); wins on three prompt classes, loses on the foliage prompt (+16%) where latent L2 improves.

**Prior art and limits.** CAT-Flow (Li et al., 2026) proposes training-free, zero-extra-NFE curvature-adaptive steps for flow matching; I found it after the experiments, read only its abstract and introduction, and did not compare. Static schedules fitted to a model: Align Your Steps, Optimal Stepsize Distillation, Bespoke Solvers. What I think this adds: the decomposition, the calibration against true local error on a real model, the noise floor, and the borrowed-grid/oracle test of per-sample adaptation. Limits: one model, 12 units, latent L2 as the main metric, the propagation profile measured on two of the evaluation units, 3-10% sensitivity of the output to 1e-7 sigma shifts in bf16.

Code, tests, tables: https://github.com/YakovSava/zero-cost-proxy-research. Write-up: https://doi.org/10.5281/zenodo.24143896.

*(Code and first drafts were produced with an AI coding assistant, Claude, under my direction; I checked the results.)*
