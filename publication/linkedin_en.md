# LinkedIn (English)

Attach: `figures/grid_face_s42.jpg`. LinkedIn shows about the first three lines before "see more", so the opening carries the post. Put the links in the first comment if you prefer the post itself without links. Create the repository and run `git push` first.

---

I tested a simple idea for choosing step sizes in image-generation samplers. Half of it held up, and the half that failed taught me more.

The idea: a sampler already has two consecutive velocity vectors, so the angle between them is a free estimate of how sharply the trajectory bends. Big angle, smaller step.

What I measured on one flow model (Anima, 12 images, against a converged reference):

- The angle is a good error predictor. Its estimate of a step's true error is consistently about 0.8 of the real value, with 0.91 correlation across images.
- As an online controller it did not help. Its result depended on its blind first step, and a step grid made for one image worked as well on a different image.
- As a fixed grid it did. Spacing the steps by equal turning angle lowered the latent error by 30 to 43 percent against the default schedule at the same number of network calls.
- The metric can disagree. On a texture-heavy prompt, perceptual distance (LPIPS) got worse for the best grid while latent distance improved.

Not new: curvature-adaptive steps without extra network calls were proposed before (CAT-Flow, arXiv 2609.01746; I found it after my experiments and did not compare), and static schedules fitted to a model exist too. What I think the study adds is the test that per-sample adaptation gives nothing over a static grid on this model, and a measurement of a bf16 noise floor that breaks naive angle controllers.

Limits: one model, 12 images, hypotheses not pre-registered, first-step finding post hoc, main metric is latent distance.

Preprint: https://doi.org/10.5281/zenodo.24143896
Code: https://github.com/YakovSava/zero-cost-proxy-research

#MachineLearning #DiffusionModels #GenerativeAI #NumericalMethods #OpenSource #ComfyUI

*(Optional: code and first drafts produced with an AI coding assistant, Claude, under my direction; I checked the results.)*
