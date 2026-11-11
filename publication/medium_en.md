# A Free Curvature Signal for Flow-Matching Samplers: It Predicts the Error, but a Static Step Grid Does the Work

*Subtitle: The angle between two consecutive velocities estimates the local error without extra network calls. On Anima it works as a predictor. As a controller it adds nothing over a fixed grid built from the same signal, and the step grid you choose matters more than the sampler.*

**Tags (Medium allows five):** Machine Learning, Generative AI, Stable Diffusion, Open Source, Numerical Methods

*Images to upload from the repository's `figures/` folder: `grid_face_s42.jpg` (hero), `error_vs_first_step.png`, `monitor_profiles.png`, `grid_fur_s42.jpg`. Create the repository and run `git push` before publishing so the links work. The short AI-assistance note at the end is optional; edit or remove it.*

---

A flow-matching image model draws a picture by integrating an ordinary differential equation, and every step costs a pass through a large network. A multistep sampler such as Adams-Bashforth already keeps the previous velocity in memory. The cosine between the old and the new velocity is therefore free. If it is close to 1 the path is straight and the step can grow. If the direction turns sharply, the step should shrink.

I wanted to turn that into a proper framework and then test it on a real model. The result was less tidy than the idea, so this is partly a story about which half of it survived.

## Not a new idea

First the caveat. A training-free, zero-extra-evaluation, curvature-adaptive step size for flow matching has already been proposed: CAT-Flow (Li et al., arXiv 2609.01746) estimates curvature from a finite difference of the vector field over time and reports up to 40% fewer steps on four text-to-image models. I found it after my experiments were done, read only its abstract and introduction, and did not compare against it. Static step schedules fitted to a model or to reference trajectories also exist (Align Your Steps, Optimal Stepsize Distillation, Bespoke Solvers). What follows is a measurement study on one model, and I will say where I think it adds something.

## What the cosine sees, and what it does not

For any two vectors, the squared difference splits exactly into a change of length and a rotation:

`|v_n - v_(n-1)|^2 = (s_n - s_(n-1))^2 + 4 s_n s_(n-1) sin^2(theta/2)`

The cosine knows only the second term. A flow can be perfectly straight and still accelerate along the line, and the cosine will not notice. A single Gaussian shows the opposite problem: its velocity flips sign at the variance minimum, the cosine reads -1, and the local error is tiny.

How much does this matter on a real model? On Anima, speed change carries 0.3 to 0.4 percent of the acceleration energy through most of the trajectory, and 8 to 9 percent (up to 23) in the first and last 5 percent of the time. In the bulk, the cosine is nearly the whole story. At the ends it is not.

## It is a good error predictor

I computed the true local Euler error from a converged reference trajectory and compared it with `E = (h/2) |v| theta`, where theta is the angle between consecutive velocities. Over 12 units (4 prompts, 3 seeds) and steps from 0.01 to 0.12 of the time interval, the median ratio of estimate to truth is 0.80 to 0.84, and across units the log-correlation is 0.91 to 0.92. A calibration constant is enough. That part of the proposal holds.

## A noise floor hides in the velocities

Anima runs in bf16 with classifier-free guidance. The velocity there has an angular noise of about 1.7 degrees between consecutive evaluations, and about 3.4 degrees in the bulk of a long trajectory. It does not shrink with the step. I tested where it comes from: the evaluation is deterministic, and perturbing the input by a relative 1e-5 already changes the direction by 1.7 degrees. The effect stays flat up to 1e-3. That is the signature of rounding, not of a rough flow. Without guidance the floor is roughly four times lower.

A controller that holds the angle per step at a target below this floor cannot reach its target and collapses to its minimum step. A related trap: the norm of a derivative estimated from noisy values in 110,000 dimensions is biased upward by the noise power, so a raw finite difference shows a plateau that is not there. Removing the residual variance of a local polynomial fit fixes it.

## The first controller lost, and the reason was the first step

I built the controller as proposed: `h_next = h * (target_angle / measured_angle)`, clamped. Its error at equal NFE was worse than the default schedule, and strangely it did not depend on the target angle or on how many evaluations it used. It depended on the initial step: mean error 0.80 for an initial step of 0.06, 0.51 for 0.02, and 0.26 for 0.01. At the first step there is no history, so the controller is blind exactly where the curvature is largest.

## A static grid built from the same signal wins

If the problem is where the steps go, I can build the grid offline. I took the mean angle rate `|v'|/|v|` of the other 11 units, set the step density proportional to it (equal turning angle per step), and ran plain Euler.

Against the default ComfyUI schedule at the same number of evaluations, the error to a converged solution is lower by 30 to 43 percent for 8, 12, 16 and 24 evaluations, with every interval excluding zero. Weighting by a measured error-propagation profile (errors made early weigh about four times more at the end) takes it to 41 to 50 percent, and 8 evaluations then give a lower error than the default at 24. The textbook exponent for Euler, one half, is clearly better than the default only at 24 evaluations. In the 8 to 24 step regime the problem is far from asymptotic.

![face_s42: default schedule (top) and the static grid (bottom) at 8, 12, 16 steps](grid_face_s42.jpg)

On the face prompt the default schedule draws a different composition at 8, 12 and 16 steps. The static grid reaches the composition of the converged image at 8.

## Per-sample adaptation added nothing I could measure

Two direct tests. First, I took the step grid the controller produced for one image and applied it to a different image. The error was the same as with the image's own grid (log-ratio -0.01 to -0.02, interval about plus or minus 0.07). Second, the controller with a small initial step ended up equal to the static grid it implies (log-ratio -0.02, interval -0.09 to +0.04). The grid's shape carries the gain. The adaptation to the individual trajectory carries no information I could detect. A comparison of a controller with a fixed default schedule cannot show this, because the default is a bad grid. I do not know what the same test would give for other controllers.

## It depends on the metric

I checked the best grid with LPIPS against the converged image. Over 8, 12 and 16 evaluations the LPIPS is 30 to 35 percent lower on average (it wins on 11 of 12, 9 of 12 and 9 of 12 units). By prompt it wins on flat colour, faces and geometric grids (39 to 48 percent lower) and loses on the texture-heavy prompt (about 16 percent higher), even though the latent distance there improves.

![fur_s42: the static grid is smoother and flatter](grid_fur_s42.jpg)

The grid puts its steps early, where composition is decided, and few at the end, where fine texture appears. It aligns the low frequencies better and loses texture energy. A grid for texture-heavy content would need a perceptual weight, and I did not build one.

## What I would not claim

* One model (Anima), one resolution, one guidance scale, 12 units. Effects below roughly 5 to 8 percent are not detectable.
* The main metric is the latent distance to a converged solution with an own error of about 5 percent. LPIPS was checked for one grid only.
* The hypotheses were not pre-registered in a file before the measurements. The first-step finding came from a result that looked like a bug, so it is post hoc. I designed the control experiments afterwards.
* The propagation profile used by the best grid was measured on two of the 12 evaluation units. The plain angle-rate grids are leave-one-out.
* Euler and Adams-Bashforth 2 only. The rapid step growth of these grids at the start makes Adams-Bashforth 2 worse than Euler.
* Shifting the sigmas by a relative 1e-7 changes the final latent by 3 to 10 percent in bf16. The numbers are deterministic for the code, but there is scatter of several percent per unit between implementations.

## Using it

The repository has the code, 30 tests, all tables and figures, and sigma lists for Anima at 8, 12, 16 and 24 evaluations that you can feed to a manual-sigmas node with the stock `euler` sampler (576x768, CFG 5, Anima only). There is no ComfyUI node. To make a grid for another model you need reference trajectories of that model to compute the angle-rate profile, which takes about two minutes per image here. I did not test how few images are enough.

Full record, hypothesis table and all intervals: the README in the repository. Preprint and code: links to be added after publication.

*AI assistance (optional): the code, analysis scripts and first drafts of this text were produced with an AI coding assistant (Claude, Anthropic) under my direction. I checked the results and I am responsible for the content.*
