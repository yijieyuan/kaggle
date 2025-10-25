# Approach Overview and Key Learnings

**This is a rough summary from long after the competition ended):**

1. Initially spent time figuring out how to run inference in Kaggle's GPU environment, such as SDXL and FLUX, which required running inference in segments and offloading to GPU.
2. Later experimented with FLUX combined with prompt engineering, but discovered that prompt engineering had limited impact—the model's capabilities matter much more.
3. Investigated better methods for converting PNG to SVG, which led to exploring the field of Primitive Drawing. Then improved upon a fast C language version and GPU version for Primitive Drawing. (Also developed this https://yijiey.com/playground/primitive-drawing/)
4. Spent considerable time researching differentiable rendering to improve the Aesthetic Score, but struggled with anti-aliasing filters. The rendered images had potential errors, and some filters like bilateral filtering were extremely slow. Ultimately didn't succeed with this approach, and also tried using genetic algorithms to optimize the Aesthetic Score without significant improvement. A differentiable approach was still needed—the third-place team eventually figured out a fully differentiable pipeline and optimized it using diff-svg.
5. Additionally, some teams managed to make text appear in post-processed images, which was the approach used by the first and second place winners.





