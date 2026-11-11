# References (APA 7)

## Что проверено, а что нет

Этот список составлен 2026-11-11 с доступом к сети. Для каждой работы я пытался открыть страницу первоисточника (arXiv, PMLR, NeurIPS/ICLR proceedings, DOI-ссылку). Глубина проверки разная, она описана по уровням в таблице ниже.

| Уровень проверки | Работы |
| --- | --- |
| **A.** Страница открыта, название и аннотация совпали, список авторов виден на странице или в BibTeX | Align Your Steps (BibTeX на PMLR, в т.ч. страницы); Bespoke Solvers; AdaptiveDiffusion; CAT-Flow; Optimal Stepsize; UniPC (arXiv, NeurIPS 2023 указан в комментарии) |
| **B.** Страница открыта, название и аннотация совпали, авторы, площадка и страницы даны по памяти или из списка в соседнем репозитории | Flow Matching; Rectified Flow; SD3; Classifier-free guidance; EDM; DPM-Solver; DPM-Solver++; SA-Solver; LPIPS; BFloat16; CAB; Minimizing Trajectory Curvature; Qwen-Image |
| **C.** Подтверждено только поиском: работа существует, DOI указан в результатах, остальное по памяти | Dormand и Prince (1980); Gustafsson и др. (1988); Söderlind (2002); Eriksson и др. (1995); de Boor (1973); Savitzky и Golay (1964); Virtanen и др. (2020); Jolicoeur-Martineau и др. (2021) |
| **D.** Не открывалось | Hairer и др. (1993); Bashforth и Adams (1883); Efron (1979); Hunter (2007); Harris и др. (2020); Paszke и др. (2019). Записи Zenodo (Savelev, 2026a-c; Savelev и Yuzvynsky, 2026): с этой машины не открылись, названия взяты из `CITATION.cff` и README соседних репозиториев |

Перед подачей на Zenodo пробегитесь по уровням B-D: номера страниц и порядок авторов там могут отличаться от того, что написано ниже.

Особые замечания:

* **Anima.** У модели есть карточка на Hugging Face (CircleStone Labs совместно с Comfy Org, 2 млрд параметров). Файл, с которым я работал, называется `anima_aestheticV11(1).safetensors`; карточка описывает версию Anima-Aesthetic, но именно эту сборку (v1.1) и источник файла я не проверял. Год публикации модели на карточке не указан.
* **Порядок авторов у Lee, Kim & Ye (2023)** в загруженной странице обрезан; имя первого автора (Sangyun Lee) дано по памяти.
* **CAT-Flow** опубликован 2026-09-01 (arXiv:2609.01746). Я нашёл его после завершения экспериментов. Прочитаны аннотация и введение, экспериментальные разделы целиком не читались.
* В `dpmpp_extended/README.md` соседнего проекта UniPC указан как arXiv:2303.01872 и Lipman et al. как «Nicklas»; правильно arXiv:2302.04867 и «Nickel» (в списке ниже исправлено).

## Список (в алфавитном порядке)

Bashforth, F., & Adams, J. C. (1883). *An attempt to test the theories of capillary action by comparing the theoretical and measured forms of drops of fluid, with an explanation of the method of integration employed in constructing the tables which give the theoretical forms of such drops*. Cambridge University Press.

CircleStone Labs. (n.d.). *Anima* [Computer software]. Hugging Face. https://huggingface.co/circlestone-labs/Anima

Comfy Org. (n.d.). *ComfyUI* [Computer software]. GitHub. https://github.com/comfyanonymous/ComfyUI

de Boor, C. (1973). Good approximation by splines with variable knots. In A. Meir & A. Sharma (Eds.), *Spline functions and approximation theory* (pp. 57-72). Birkhäuser. https://doi.org/10.1007/978-3-0348-5979-0_3

Dormand, J. R., & Prince, P. J. (1980). A family of embedded Runge-Kutta formulae. *Journal of Computational and Applied Mathematics, 6*(1), 19-26. https://doi.org/10.1016/0771-050X(80)90013-3

Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1-26. https://doi.org/10.1214/aos/1176344552

Eriksson, K., Estep, D., Hansbo, P., & Johnson, C. (1995). Introduction to adaptive methods for differential equations. *Acta Numerica, 4*, 105-158. https://doi.org/10.1017/S0962492900002531

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., & Rombach, R. (2024). Scaling rectified flow transformers for high-resolution image synthesis. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235). https://arxiv.org/abs/2403.03206

Gustafsson, K., Lundh, M., & Söderlind, G. (1988). A PI stepsize control for the numerical solution of ordinary differential equations. *BIT Numerical Mathematics, 28*(2), 270-287. https://doi.org/10.1007/BF01934091

Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving ordinary differential equations I: Nonstiff problems* (2nd rev. ed.). Springer. https://doi.org/10.1007/978-3-540-78862-1

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2

Ho, J., & Salimans, T. (2022). *Classifier-free diffusion guidance* [Preprint]. arXiv. https://arxiv.org/abs/2207.12598

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90-95. https://doi.org/10.1109/MCSE.2007.55

Jolicoeur-Martineau, A., Li, K., Piché-Taillefer, R., Kachman, T., & Mitliagkas, I. (2021). *Gotta go fast when generating data with score-based models* [Preprint]. arXiv. https://arxiv.org/abs/2105.14080

Kalamkar, D., Mudigere, D., Mellempudi, N., Das, D., Banerjee, K., Avancha, S., Vooturi, D. T., Jammalamadaka, N., Huang, J., Yuen, H., Yang, J., Park, J., Heinecke, A., Georganas, E., Srinivasan, S., Kundu, A., Smelyanskiy, M., Kaul, B., & Dubey, P. (2019). *A study of BFLOAT16 for deep learning training* [Preprint]. arXiv. https://arxiv.org/abs/1905.12322

Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *Advances in Neural Information Processing Systems, 35*, 26565-26577. https://arxiv.org/abs/2206.00364

Lee, S., Kim, B., & Ye, J. C. (2023). Minimizing trajectory curvature of ODE-based generative models. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202). https://arxiv.org/abs/2301.12003

Li, Q., Cisneros-Velarde, P., Fu, K., Miranda, S. A., Vaswani, S., & Zhang, H. (2026). *CAT-Flow: Curvature-adaptive steps for flow matching* [Preprint]. arXiv. https://arxiv.org/abs/2609.01746

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. In *International Conference on Learning Representations*. https://arxiv.org/abs/2210.02747

Liu, X., Gong, C., & Liu, Q. (2023). Flow straight and fast: Learning to generate and transfer data with rectified flow. In *International Conference on Learning Representations*. https://arxiv.org/abs/2209.03003

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022a). DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. *Advances in Neural Information Processing Systems, 35*, 5775-5787. https://arxiv.org/abs/2206.00927

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022b). *DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models* [Preprint]. arXiv. https://arxiv.org/abs/2211.01095

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*, 8024-8035.

Pei, J., Hu, H., & Gu, S. (2025). *Optimal stepsize for diffusion sampling* [Preprint]. arXiv. https://arxiv.org/abs/2503.21774

Roy, A., & Nair, P. (2026). *CAB: Accelerating flow and diffusion sampling via rectification and corrected Adams-Bashforth* [Preprint]. arXiv. https://arxiv.org/abs/2605.16736

Sabour, A., Fidler, S., & Kreis, K. (2024). Align your steps: Optimizing sampling schedules in diffusion models. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 42947-42975). https://proceedings.mlr.press/v235/sabour24a.html

Savelev, I. (2026a). *Flow-WLS: Regularized multistep integration for accelerated sampling of flow matching generative models* [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.23337579

Savelev, I. (2026b). *Flow-WLS under quantization: Low-order multistep samplers for guided and INT8 flow-matching models* (Version 1.0) [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.23939696

Savelev, I. (2026c). *Where, not how much: Fixed versus adaptive noise schedules for flow-matching SDE samplers at low NFE* [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.23857361

Savelev, I., & Yuzvynsky, Y. (2026). *Adaptive step samplers: Zero-cost truncation error estimation and exposed stochastic control for ComfyUI* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23529356

Savelev, I. (2026d). *Zero-cost curvature proxies for adaptive stepping in flow matching: A calibrated error predictor, a static grid, and no measurable headroom for per-sample control* (Version 1.0) [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.24143896

Savelev, I. (2026e). *zero-cost-proxy-research* [Computer software]. GitHub. https://github.com/YakovSava/zero-cost-proxy-research

Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry, 36*(8), 1627-1639. https://doi.org/10.1021/ac60214a047

Shaul, N., Perez, J., Chen, R. T. Q., Thabet, A., Pumarola, A., & Lipman, Y. (2024). Bespoke solvers for generative flow models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2310.19075

Söderlind, G. (2002). Automatic control and adaptive time-stepping. *Numerical Algorithms, 31*(1-4), 281-310. https://doi.org/10.1023/A:1021160023092

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., ... SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261-272. https://doi.org/10.1038/s41592-019-0686-2

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S., Bai, S., Xu, X., Chen, Y., ... Qwen Team. (2025). *Qwen-Image technical report* [Preprint]. arXiv. https://arxiv.org/abs/2508.02324 **[полный список авторов см. на arXiv; здесь сокращён]**

Xue, S., Yi, M., Luo, W., Zhang, S., Sun, J., Li, Z., & Ma, Z.-M. (2023). SA-Solver: Stochastic Adams solver for fast sampling of diffusion models. *Advances in Neural Information Processing Systems, 36*. https://arxiv.org/abs/2309.05019

Ye, H., Yuan, J., Xia, R., Yan, X., Chen, T., Yan, J., Shi, B., & Zhang, B. (2024). Training-free adaptive diffusion with bounded difference approximation strategy. *Advances in Neural Information Processing Systems, 37*. https://arxiv.org/abs/2410.09873

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 586-595). https://doi.org/10.1109/CVPR.2018.00068

Zhao, W., Bai, L., Rao, Y., Zhou, J., & Lu, J. (2023). UniPC: A unified predictor-corrector framework for fast sampling of diffusion models. *Advances in Neural Information Processing Systems, 36*. https://arxiv.org/abs/2302.04867

## Что осталось без ссылки

| Что использовано | Что нужно |
| --- | --- |
| Файл модели `anima_aestheticV11(1).safetensors`, текстовый энкодер `anima_baseV10_txt.safetensors` | Источник именно этих файлов, их версия и дата. Ссылка на карточку Anima выше общая |
| Реализация `normal` в ComfyUI (shift 3) | Версия ComfyUI; в тексте описана формула, ссылки на коммит нет |
| Пакет `lpips` 0.1.4 | Сам метод процитирован (Zhang и др., 2018); ссылку на пакет можно добавить |

## Где что цитировать

| Утверждение | Ссылка |
| --- | --- |
| Flow matching, rectified flow, SD3-подобные модели | Lipman et al. (2023); Liu et al. (2023); Esser et al. (2024) |
| Classifier-free guidance | Ho & Salimans (2022) |
| Эйлер, Адамс-Бэшфорт, оценка Хойна, вложенные методы | Hairer et al. (1993); Bashforth & Adams (1883); Dormand & Prince (1980); Karras et al. (2022) |
| Equidistribution (равнораспределение) как принцип выбора сетки | de Boor (1973); Eriksson et al. (1995) |
| Контроллер шага как линейный фильтр по log h | Gustafsson et al. (1988); Söderlind (2002) |
| Локальная полиномиальная подгонка и производные по зашумлённым данным | Savitzky & Golay (1964) |
| Ускоренные солверы | Lu et al. (2022a, 2022b); Zhao et al. (2023); Xue et al. (2023); Roy & Nair (2026) |
| Шаг по кривизне без дополнительных вызовов сети (ближайшая работа) | Li et al. (2026) |
| Пропуск шагов по разностям | Ye et al. (2024) |
| Статические расписания, оптимизированные по данным или по траекториям | Sabour et al. (2024); Pei et al. (2025); Shaul et al. (2024); Esser et al. (2024); Karras et al. (2022) |
| Адаптивные солверы для генеративных моделей | Jolicoeur-Martineau et al. (2021) |
| Кривизна траекторий и ошибка усечения | Lee et al. (2023) |
| bf16 как источник шума округления | Kalamkar et al. (2019) |
| LPIPS | Zhang et al. (2018) |
| Бутстреп | Efron (1979) |
| Программная среда | Paszke et al. (2019); Harris et al. (2020); Virtanen et al. (2020); Hunter (2007); Comfy Org (n.d.) |
| Модель и VAE | CircleStone Labs (n.d.); Wu et al. (2025) |
| Прежние работы автора | Savelev & Yuzvynsky (2026); Savelev (2026a, 2026b, 2026c) |
