---
title: Diffusers Preview Demo
emoji: 📉
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 3.3.1
app_file: app.py
pinned: false
---

Here we demonstrate how to do preview images of a Stable Diffusion's intermediate stages using a [fast approximation][approximation] to visualize the low-resolution (64px) latent state.

+ **[app.py](app.py)** is a Gradio application that yields preview images from a [generator function][gradio.iter] while the pipeline is in progress. The UI is directly derived from Stability AI's [Stable Diffusion Demo](https://huggingface.co/spaces/stabilityai/stable-diffusion).
+ **[progress_ipywidgets_demo.ipynb](progress_ipywidgets_demo.ipynb)** demonstrates using the same pipeline to update [Jupyter widgets][ipywidgets] in a notebook.
+ [`preview_decoder.py`](preview_decoder.py) has the fast latent-to-RGB decoder function.
+ [`generator_pipeline.py`](generator_pipeline.py) provides a DiffusionPipeline with a `generate()` method to yield the latent data at each step. It is nearly a strict refactoring of the [StableDiffusionPipeline][sdpipeline] in 🧨diffusers 0.3.

[approximation]: https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2?u=keturn "Decoding latents to RGB without upscaling"
[ipywidgets]: https://ipywidgets.readthedocs.io/en/stable/
[gradio.iter]: https://gradio.app/key_features/#iterative-outputs
[sdpipeline]: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
