---
title: "Chapter-2 Deep Learning for Coders with FastAI & Pytorch"
date: 2021-06-23
slug: ch2-dl-for-coders
category: Deep Learning
summary:
description: "From Model to Production"
cover:
  image:
  alt:
  caption: 
  relative: true
showtoc: true
TocOpen: false
draft: false
params:
  ShowReadingTime: true
  ShowShareButtons: false
  ShowBreadCrumbs: true
---

### Keypoints

- For deploying models into production we need data, obviously a trained model, API’s around Model allowing it to be accessed by the users, nice UI/UX experience if the model will be served directly as a service from the browser, good infrastructure, best coding practices etc.
- There are 4 main categories in a deep learning project before production.
  - Data Preparation
  - Labelling data
  - Model Training
  - Production
- The better way would be to allocate same time for each task.
- Underestimating the constraints and overestimating the capabilities of deep learning may lead to frustratingly poor results. So be keen on understanding what is needed.
- Conversely, overestimating the constraints and underestimating the capabilities of deep learning may mean you do not attempt a solvable problem because you talk yourself out of it. So don’t stop yourself from trying a model. Iterate your learnings.
- It’s better to iterate the project end-to-end rather than just fine-tuning the model or making some fancy GUI.
- It’s only by practicing (and failing) a lot that you will get an intuition of how to train a model.
- Start learning with the existing examples and the existing domains where deep learning is already applied and then look for more branches.
- There are many accurate models that are of no use to anyone, and many inaccurate models that are highly useful.
- A Drivetrain approach of how to use data not just to generate data but to produce actionable results is shown in the below picture:

![drive-train](/blogs/b2/drive-train.png)

- Below is the cool little gist which shows right from how to make our datasets to training & inference.
<cite>Gist here[^1]</cite>
[^1]: [Example Gist](https://gist.github.com/RaviChandraVeeramachaneni/12b2ed5ef7342048f92a86b019d4fd2f)
- Some of the problems to understand while building data centric products with deep learning involved:
  - Understanding and testing the behavior of a deep learning model is much more difficult than with most other code we write.
  - The neural network’s behavior emerges from the model’s attempt to match the training data, rather than being exactly defined. So this could be a disaster.
  - Out-of-domain data and domain shifts are another problem to be considered.
  - One possible approach outlined to understand the problems would be best described by below Image.

  ![data-centric-app](/blogs/b2/data-centric-app.png)

  ### Appendix