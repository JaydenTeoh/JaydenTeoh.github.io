---
title: "Mathematics behind Diffusion Models"
date: "2025-08-16"
tags: ["machine learning", "research"]
excerpt: "Personal notes during my journey of understanding diffusion models."
coverImage: "blog/assets/2025-08-16-diffusion-models/Untitled%203.png"
---

# Mathematics behind Diffusion Models

## What are Diffusion Models?

The concept of diffusion-based generative modelling was actually proposed early in 2015 by [Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585) who were inspired by non-equilibrium thermodynamics. The idea behind it as described by Sohl-Dickstein et al. (2015) is this:

> The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data.
> 

Half a decade later, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) proposed Denoising Diffusion Probabilistic Models (DDPMs), which improved upon the previous method by introducing significant simplifications to the training process.  Soon after, a [2021 paper by OpenAI](https://arxiv.org/pdf/2105.05233) demonstrated DDPMs’ superior performance in image synthesis tasks compared to [Generative Adversarial Networks (GANs)](https://www.notion.so/Generative-Adversarial-Networks-ebfba7327d264466b8496a52cf98e453?pvs=21). Since then, notable diffusion-based generative models have been released such as [DALL-E](https://arxiv.org/abs/2102.12092), [Stable Diffusion](https://arxiv.org/abs/2112.10752) and [Imagen](https://arxiv.org/abs/2205.11487). I’ll be covering the concept underlying diffusion models, mainly focusing on DDPMs.

To better understand this, let’s focus on the forward diffusion process and reverse diffusion process separately.

### Forward Diffusion Process

![Image by [Karagiannakos and Adaloglou (2022)](https://theaisummer.com/diffusion-models/), modified from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)](../assets/2025-08-16-diffusion-models/Untitled.png)

In the forward trajectory, we want to gradually “corrupt” the training images. As such, we iteratively apply Gaussian noise to images sampled from the true data distribution, i.e. $x_0 \sim q(x)$, in over $T$ steps to produce a sequence of noisy samples $x_0, x_1, \dots , x_T$. 

The diffusion process is fixed to a Markov chain which simply means that each step is only dependent on the previous one (memoryless). Specifically, at each step, we apply Gaussian noise with variance $\beta_T \in (0,1)$ to $x_{t-1}$ to produce a latent variable $x_t$ of the same dimension. As such, each transition is parameterized as a diagonal Gaussian distribution that uses the output of the previous state as its mean:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})
$$

Note:

- $\beta_T$ is known as the “diffusion rate” and it can be sampled according to a variance schedule $\beta_1, \dots, \beta_T$, which means the amount of noise applied at each time step is not necessarily constant
- $\mathbf{I}$ is the identity matrix. We use the identity matrix because our images are multi-dimensional and we want each dimension to be independent of each other

The posterior after $T$ steps, conditioned on the original data distribution, can be represented as a product of single step conditionals as such:

$$
q(x_{1:T} | x_0) = \prod^T_{t=1} q(x_t | x_{t-1})
$$

As $T\rightarrow \infty$, $x_T$ is equivalent to an isotropic Gaussian distribution, losing all information about the original sample distribution.

### Reverse Diffusion Process

![Image by [Karagiannakos and Adaloglou (2022)](https://theaisummer.com/diffusion-models/), modified from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)](../assets/2025-08-16-diffusion-models/Untitled%201.png)

In the reverse process, we aim to learn a model that can denoise the pure Gaussian noise, i.e. $x_T \sim \mathcal{N}(x_T; 0,\mathbf{I})$, to recover the original sample image. As mentioned by Sohl-Dickstein et al. (2015),

> Estimating small perturbations is more tractable than explicitly describing the full distribution with a single, non-analytically-normalizable, potential function.
> 

Directly describing the original distribution from the pure Gaussian noise can be intractable. Rather, what we can do is train a model $p_\theta$ (e.g. using a neural network) to approximate $q(x_{t-1}|x_{t})$ such that we can iteratively recover the original data distribution in small time steps. Therefore, the reverse trajectory can also be formulated as a Markov chain and can be represented as such:

$$
p_\theta(x_{0:T}) = p(x_T) \prod^T_{t=1} p_\theta(x_{t-1} | x_t)
$$

where $p(x_T) = \mathcal{N}(x_T; 0,\mathbf{I})$.

Moreover, since $q(x_t|x_{t-1})$ follows a Gaussian distribution, if $\beta_t$ is small, then the reversal of the diffusion process has the identical functional form as the forward process [(Feller, 1949)](https://www.semanticscholar.org/paper/On-the-Theory-of-Stochastic-Processes%2C-with-to-Feller/4cdcf495232f3ec44183dc74cd8eca4b44c2de64), which means that $q(x_{t-1}|x_{t})$ will also be Gaussian. Therefore, to approximate $q(x_{t-1}|x_{t})$, our model $p_\theta$ only needs to estimate the Gaussian parameters $\mu_\theta(x_t, t)$ and $\Sigma_\theta(x_t, t)$ for timestep $t$.

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

Note:

- $p_\theta$ not only takes in $x_t$, but also $t$, as inputs because each time step is associated with different noise levels.

## Parameterisation of Diffusion Model

In summary, we have defined the forward trajectory as a steady noisification of the sample distribution over time and the reverse trajectory as “tracing back” these steps to recover the original distribution. 

But how exactly do we teach a neural network (or other function approximation methods) to approximate the conditional probabilities for each time step in the reverse trajectory? To do so, we need to define a loss function. 

Naively, we can use a maximum likelihood objective where we maximize the likelihood assigned to $x_0$ by the model, i.e.

$$
\begin{aligned}
p_\theta(x_0) &= \int p_\theta(x_{0:T})dx_{1:T}  \\
L &= -\log(p_\theta(x_0))
\end{aligned}
$$

This objective is unfortunately intractable as it requires us to marginalize over all possible trajectories we could have taken from $x_{1:T}$. Rather, we can take inspiration from Variational Autoencoders (VAE) and reformulate the training objective using a variational lower bound (VLB), also known as **“evidence lower bound” (ELBO)**.

$$
\begin{aligned}\log p_\theta(x_0)
&\geq \log p_\theta(x_0) - D_\text{KL}(q(x_{1:T} | x_0) \| p_\theta(x_{1:T} | x_0) ) \\
&= \log p_\theta(x_0) - \mathbb{E}_{q(x_{1:T} | x_0)} \Big[ \log\frac{q(x_{1:T} | x_0)}{\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}} \Big] \\
&= \log p_\theta(x_0) - \mathbb{E}_{q(x_{1:T} | x_0)} \Big[ \log\frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0) \Big] \\
&= - \mathbb{E}_{q(x_{1:T} | x_0)} \Big[ \log\frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}  \Big]
\end{aligned}
$$

Therefore, the last term becomes the VLB of the likelihood assigned to $x_0$, a proxy objective to maximize. However, this VLB term is still not tractable so further reformulations is needed. Before we proceed, it is important to note that we can rewrite each transition as $q(x_{t}|x_{t-1}) = q(x_{t}|x_{t-1}, x_0)$, where the extra conditioning term is superfluous due to the Markov property. Using Bayes’ rule, we can rewrite each transition as:

$$
q(x_{t}|x_{t-1}, x_0) = \frac{q(x_{t-1}|x_{t}, x_0)q(x_{t}|x_0)}{q(x_{t-1}|x_0)}
$$

This trick will be useful for reducing the variance and derive a more elegant variational lower bound expression. Continuing from where we left off earlier, we have:

$$
\begin{aligned}\log p_\theta(x_0)
&\geq\mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})\prod_{t=1}^T p_\theta(x_{t-1} | x_{t})}{\prod_{t=1}^T q(x_{t} | x_{t-1})}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)\prod_{t=2}^T p_\theta(x_{t-1} | x_{t})}{q(x_1 |x_0)\prod_{t=2}^T q(x_{t} | x_{t-1})}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)\prod_{t=2}^T p_\theta(x_{t-1} | x_{t})}{q(x_1 |x_0)\prod_{t=2}^T q(x_{t} | x_{t-1}, x_0)}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{q(x_1 |x_0)} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{q(x_{t} | x_{t-1}, x_0)}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{q(x_1 |x_0)} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{\frac{q(x_{t-1}|x_{t}, x_0)q(x_{t}|x_0)}{q(x_{t-1}|x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{q(x_1 |x_0)} + \log\prod_{t=2}^T \frac{\cancel{q(x_{t-1}|x_0)}}{\cancel{q(x_{t}|x_0)}} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{\cancel{q(x_1 |x_0)}} + \log \frac{\cancel{q(x_{1}|x_0)}}{{q(x_{T}|x_0)}} + \log\prod_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\log \frac{p_\theta(x_{T})p_\theta(x_0 | x_1)}{{q(x_{T}|x_0)}} + \log\sum_{t=2}^T \frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1:T} | x_0)}[\log p_\theta(x_0 | x_1)] + \mathbb{E}_{q(x_{1:T} | x_0)}\Big[\log \frac{p_\theta(x_{T})}{{q(x_{T}|x_0)}} \Big] + \log\sum_{t=2}^T \mathbb{E}_{q(x_{1:T} | x_0)} \Big[\frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \mathbb{E}_{q(x_{1} | x_0)}[\log p_\theta(x_0 | x_1)] + \mathbb{E}_{q(x_{T} | x_0)}\Big[\log \frac{p_\theta(x_{T})}{{q(x_{T}|x_0)}} \Big] + \log\sum_{t=2}^T \mathbb{E}_{q(x_{t}, x_{t-1} | x_0)} \Big[\frac{p_\theta(x_{t-1} | x_{t})}{{q(x_{t-1}|x_{t}, x_0)}}  \Big] \\
&= \underbrace{\mathbb{E}_{q(x_{1} | x_0)}[\log p_\theta(x_0 | x_1)]}_{\text{reconstruction term}} - \underbrace{D_\text{KL} (q(x_{T}|x_0) \parallel p_\theta(x_{T}))}_{\text{prior matching term}}  -\sum_{t=2}^T \mathbb{E}_{q(x_{t} | x_0)} [\underbrace{D_\text{KL} ({q(x_{t-1}|x_{t}, x_0)} \parallel p_\theta(x_{t-1} | x_{t}))}_{\text{denoising matching term}}]   \\
\end{aligned}
$$

- $D_\text{KL} (q(x_{T}|x_0) \parallel p_\theta(x_{T}))$ is a constant because $q$ has no trainable parameters and $p_\theta(x_T)$ is a standard Gaussian. Therefore, we can ignore it.
- $\mathbb{E}_{q(x_{1} | x_0)}[\log p_\theta(x_0 | x_1)]$ is the reconstruction term which predicts the log likelihood of the original data sample given the first-step latent. Since the original data distribution may not be Gaussian, we cannot compute this in closed form. We can approximate and optimise this term using a Monte Carlo estimate.
- $D_\text{KL} ({q(x_{t-1}|x_{t}, x_0)} \parallel p_\theta(x_{t-1} | x_{t}))$ measures the KL divergence between the learnt transition step $p_\theta(x_{t-1} | x_{t})$ and the ground-truth denoising transition step $q(x_{t-1}|x_{t}, x_0)$. $q(x_{t-1}|x_{t}, x_0)$ can act as a ground-truth signal because it defines how to denoise a noisy image $x_t$ with access to what the final, completely denoised image $x_0$ should be.

Given that the denoising matching term is the only term we are interested in, to maximise the likelihood objective, we will need to minimise the KL divergence between the learnt denoising step and ground-truth denoising step. By Bayes’ Rule, we can reformulate the ground-truth denoising transition step as such:

$$
q(x_{t-1}|x_{t}, x_0) = \frac{q(x_{t}|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}| x_0)}
$$

Using our definition of the forward transition step and the Markov property, we already know that $q(x_{t}|x_{t-1}, x_0) = q(x_{t}|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$. Let $\alpha_t = 1 - \beta_t$, under the reparameterisation trick used in VAEs, samples $x_t \sim q(x_t|x_{t-1})$ can be rewritten as:

$$
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1- \alpha_t}\epsilon
$$

where $\epsilon \sim \mathcal{N}(\epsilon; 0, \mathbf{I})$. To express $q(x_{t}| x_0)$ in closed form, we recursively apply the reparameterisation trick. That is, for any $x_t \sim q(x_t | x_0)$,

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1- \alpha_t}\epsilon^*_{t-1} \\
&= \sqrt{\alpha_t a_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t a_{t-1}}\epsilon^*_{t-2} + \sqrt{1-\alpha_t}\epsilon^*_{t-1} \\
&=  \sqrt{\alpha_t a_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t a_{t-1} + 1 - \alpha_t}\epsilon_{t-2} \\
&= \dots \\
&= \sqrt{\prod_{i=1}^t a_i}x_0 + \sqrt{1-\prod_{i=1}^t a_i}\epsilon_0 \\
&= \sqrt{\bar{a}_t}x_0 + \sqrt{1-\bar{a}_t}\epsilon_0 \\
&\sim \mathcal{N}(x_t; \sqrt{\bar{a}_t}x_0, (1-\bar{a}_t)\mathbf{I}) \\
\end{aligned}
$$

where $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$. We can merge two Gaussians in (1) since the sum of two independent Gaussian random variables is a Gaussian with mean being the sum of the two means and variance being the sum of the two variances. We have therefore derived $q(x_{t}| x_0)$ and we can reuse the parameterization trick to yield $q(x_{t-1}| x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$. Substituting both expressions into the Bayes rule expansion of the ground truth denoising step (intermediate steps have been omitted for brevity):

$$
\begin{aligned}
q(x_{t-1}|x_{t}, x_0)
&= \frac{q(x_{t}|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_{t}| x_0)}\\
&= {\frac{\mathcal{N}(x_{t} ; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)\textbf{I})\mathcal{N}(x_{t-1} ; \sqrt{\bar\alpha_{t-1}}x_0, (1 - \bar\alpha_{t-1}) \textbf{I})}{\mathcal{N}(x_{t} ; \sqrt{\bar\alpha_{t}}x_0, (1 - \bar\alpha_{t})\textbf{I})}}\\
&\propto {\text{exp}\left\{-\left[\frac{(x_{t} - \sqrt{\alpha_t} x_{t-1})^2}{2(1 - \alpha_t)} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0)^2}{2(1 - \bar\alpha_{t-1})} - \frac{(x_{t} - \sqrt{\bar\alpha_t} x_{0})^2}{2(1 - \bar\alpha_t)} \right]\right\}}\\
&= \dots \\
&\propto {\mathcal{N}(x_{t-1} ;} \underbrace{{\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}}}_{\mu_q(x_t, x_0)}, \underbrace{{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}\textbf{I}}}_{\bm{\Sigma}_q(t)})
\end{aligned}
$$

We have therefore shown that at each step, $x_{t-1} \sim q(x_{t-1}|x_{t}, x_0)$ follows a normal distribution with mean $\mu_q(x_t, x_0)$, a function of $x_t$ and $x_0$, and variance $\Sigma_q(t)$, a function of $\alpha$ coefficients. We can further leverage the reparameterization trick to express $x_0$ as $\epsilon_0$:

$$
\begin{aligned}
\mu_q(x_t, x_0) &= {\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}} \\
&= {\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}}}{1 -\bar\alpha_{t}}} \\
&= \dots \\
&= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_0
\end{aligned}
$$

Following [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), we can set $\Sigma_q(t)$  as a constant at each timestep by modelling the $\alpha$ coefficients as fixed hyperparameters. With that, we can rewrite the variance equation as $\Sigma_q(t) = \sigma^2_q(t)\mathbf{I}$, where:

$$
\sigma^2_q(t) = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}
$$

As we have kept the variance constant, minimizing the KL divergence is simply minimizing the difference between $\mu_q(x_t, x_0)$ and $\mu_\theta(x_t, t)$. Note that we have no choice but to only parameterize the mean of the learnt model $\mu_\theta(x_t, t)$ as a function of $x_t$ since $p_\theta(x_{t-1}|x_t)$ does not depend on $x_0$ (due to Markov property). 

The true denoising transition mean is expressed as:

$$
\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_0
$$

To optimize the model’s mean $\mu_\theta(x_t, t)$, we set it to have the following form:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\epsilon_\theta}(x_t, t)
$$

where $\hat{\epsilon}_\theta(x_t, t)$ is parameterized by a neural network that seeks to predict the source noise $\epsilon_0 \sim \mathcal{N}(\epsilon; 0, \mathbf{I})$ that lead to $x_t$ from $x_0$. As such, the optimization problem simplifies to:

$$
\begin{aligned}
&\quad \argmin_{{\theta}} D_\text{KL}({q(x_{t-1}|x_t, x_0)}\parallel{p_{{\theta}}(x_{t-1}|x_t)}) \\
&= \argmin_{{\theta}} D_\text{KL}({\mathcal{N}\left(x_{t-1}; {\mu}_q,{\Sigma}_q\left(t\right)\right)} \parallel {\mathcal{N}\left(x_{t-1}; {\mu}_{{\theta}},{\Sigma}_q\left(t\right)\right)})\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert\frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\hat\epsilon}_{{\theta}}(x_t, t) -
\frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\epsilon}_0\right\rVert^2_2\right]\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\epsilon}_0 - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}{\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2_2\right]\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\left[\left\lVert \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}({\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t))\right\rVert^2_2\right]\\
&=\argmin_{{\theta}}\frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{(1 - \bar\alpha_t)\alpha_t}\left[\left\lVert{\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2_2\right]
\end{aligned}
$$

We can formulate our variational lower bound loss function as

$$
\begin{aligned}
L &= \mathbb{E}_{x_0, \epsilon} \Big[ \frac{(1 - \alpha_t)^2}{2\sigma_q^2(t)(1 - \bar\alpha_t)\alpha_t}\left \lVert{\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2_2 \Big] \\
&= \mathbb{E}_{x_0, \epsilon} \Big[ \frac{\beta_t^2}{2\sigma_q^2(t)(1 - \bar\alpha_t)\alpha_t}\left \lVert{\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2_2 \Big] \tag{2}
\end{aligned}
$$

### Simplification of Loss Term

Empirically, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) found that a simplified loss function without the weighting term performs better:

$$
 L_{\text{simple}}  = \mathbb{E}_{x_0, \epsilon} \Big[\left \lVert{\epsilon}_0 - {\hat\epsilon}_{{\theta}}(x_t, t)\right\rVert^2_2 \Big]
$$

which is basically just the **mean squared error between the noise added in the forward process and the noise predicted by the model**!

![Training and sample algorithm from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)](../assets/2025-08-16-diffusion-models/Untitled%202.png)

## Relation to Score Matching

So far, I have been covering generative models via the “likelihood-based” objective, which seeks to learn a model that assigns a high likelihood to the observed data samples. However, there is another “score-based” interpretation which seeks to model the gradient of the log probability density function, a quantity known as the ***(Stein)** **score function***. Score-based generative modelling is another rabbit hole on its own, but my goal here today is just to highlight the parallels between denoising diffusion generative modelling and denoising score matching in score-based generative modelling. It would be helpful to understand this before getting into conditional diffusion models in the later section.

### Motivation behind Score Matching

In order to build a likelihood-based generative model, given a dataset $x = \{x_1, x_2, \dots, x_N\}$, one learns a function $f_\theta (x) \in \mathbb{R}$ parameterized by a learnable parameter $\theta$ that best explains the observed data. More specifically, our goal would be to find the $\theta$ that maximises the log probability density function (or probability mass function in the discrete case) of the data:

$$
\max_\theta\sum_{i=1}^N\log p_\theta(x_i)
$$

As you know from Probability Theory 101, for $p_\theta (x)$ to be a valid p.d.f., we need to make sure it is normalized such that $\int p_\theta (x) \, dx = 1$. We can define a valid p.d.f. via

$$
p_\theta (x) = \frac{e^{-f_\theta(x)}}{Z_\theta} \tag{3}
$$

where $Z_\theta = \int e^{-f_\theta(x)}dx$ is the normalizing constant and $f_\theta (x)$ in this case is often called the energy function. 

The fundamental limitation behind likelihood-based models is that can be intractable to compute the normalizing constant $Z_\theta$, especially if $x$ is usually high-dimensional and $f_\theta (x)$ is highly complex and nonlinear. One way to avoid computing $Z_\theta$ is via approximation methods. For example, in DDPMs, we relied on variational inference and used the ELBO as a surrogate objective to maximise the log likelihood of the data. 

Score-based models are motivated by this fundamental limitation and offers an alternative that avoids calculating or approximating the normalization constant by learning the score function$\nabla_x \log p_\theta (x)$ of the distribution instead. By taking the derivative of the log of the unnormalized density in Equation 3, we can see that:

$$
\begin{aligned}
\nabla_x \log p_\theta (x) &= \nabla_x \log\frac{e^{-f_\theta(x)}}{Z_\theta} \\
&= \underbrace{\nabla_x \log\frac{1}{Z_\theta}}_{=0} + \nabla_x \log e^{-f_\theta(x)} \\
&= -\nabla_x f_\theta(x) \\
&\approx s_\theta(x)
\end{aligned}
$$

where $s_\theta(x)$ is our score-based model that learns $\theta$ that best approximates $\nabla_x \log p (x)$. 

It is intuitive to see that $s_\theta(x)$ is independent of the normalization constant $Z_\theta$. And to train score-based models, we can simply minimize the expected Fisher Divergence between the $s_\theta (x)$ and the score function:

$$
\begin{aligned}
\hat{\theta} &= \argmin_{\theta} \frac{1}{2}\mathbb{E}_{p(x)} \Big[ \|  \nabla_x \log p(x) -  s_\theta(x)  \|^2_2 \Big] \tag{4}
\end{aligned}
$$

Intuitively, the Fisher Divergence measures the squared $\ell_2$ distance between the ground truth score function and the score-based model. This is what is known as ***score matching***. Instead of directly maximizing the likelihood function, score matching instead tries to find a $\theta$ such that the gradient of the model’s log likelihood is approximately the same as the gradient of the data distribution’s log likelihood, circumventing the need to work with the normalization constant.

However, a problem here is that it requires access to $\nabla_x \log p(x)$ dependent on an unknown $p(x)$, which is the same limitation faced by likelihood-based methods. Fortunately, [Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html) showed that by applying multivariate integration by parts, the objective in Equation 4 can be rewritten to not rely on $p(x)$ as such:

$$
\begin{aligned}
\hat{\theta} &= \argmin_{\theta}\mathbb{E}_{p(x)} \left[ \text{tr}({\nabla_x s_\theta(x)}) + \frac{1}{2} \|s_\theta(x)\|^2_2 \right] \tag{5}
\end{aligned}
$$

where $\nabla_x s_\theta (x)$ denotes the Jacobian of $s_\theta (x)$, which is also the Hessian of the log-density function, and $\text{tr}(\cdot)$ denotes the trace of the Jacobian matrix, i.e. the sum of the elements on the main diagonal of the matrix. 

### Intuitive Interpretation of Score Matching

We can expect the second term, $\|s_\theta(x)\|^2_2 = \| \nabla_x \log p_\theta (x)\|^2_2$, to be small when the gradients of the score function are close to zero, indicating a local extremum of the log-likelihood. This term is closely related to the maximization of the non-normalized log-likelihood. However, optimizing solely the non-normalized log-likelihood could lead to a trivial solution where the score function is flat, i.e., it has infinite variance and assigns equal probability to all $x$. Such a solution is undesirable because it fails to uniquely describe the observed data distribution.

The first term, $\text{tr}({\nabla_x s_\theta(x)})$, provides information about the overall curvature or sharpness of the extremum. To minimize the Fisher Divergence, we need this term to be negative, indicating that the extremum is a maximum. A more negative trace of the Hessian suggests a sharper maximum of the log-likelihood for the observed data points, $x$, as opposed to a flat maximum. Therefore, the trace term favors steeper maxima that more uniquely reflects the data distribution, compared to a flat maxima where similar probabilities are assigned to all data points. In some sense, this term acts as a proxy for the normalization constant in traditional density estimation. It encourages the learned score function to behave like the gradient of a properly normalized log-density function by penalizing score functions that would correspond to optimizing only the non-normalized density function.

### Langevin Dynamics

Okay, now that we have trained a score-based model $s_\theta (x) \approx \nabla_x \log p(x)$, how do we even generate samples using it? 

![Image from [Luo, C. (2022)](https://arxiv.org/pdf/2208.11970)](../assets/2025-08-16-diffusion-models/Untitled%203.png)

Intuitively, what the model has learnt is a gradient vector field over the density function of $x$ and each point in the field indicates the direction and rate of fastest increase (represented by the arrows in the image above). Langevin dynamics provide a Markov Chain Monte Carlo procedure to sample from $p(x)$ with access to only $\nabla_x \log p(x)$. Starting from any arbitrary point $x_0 \sim \pi(x)$ (represented by blue dots in image above) sampled from some assumed prior distribution $\pi$ (e.g. Gaussian), we can iteratively perform the following updates to climb towards the modes of the distribution (represented by red lines in image above):

$$
x_{t+1} \leftarrow x_t + \epsilon \nabla_x \log p(x) + \sqrt{2\epsilon}z_t,  \quad t=0,1,\dots,K
$$

where $z_t \sim \mathcal{N}(0,I)$. The addition of the Gaussian noise helps to introduce stochasticity to the generative process, allowing the samples to converge towards different modes other than the nearest. When $\epsilon$ is sufficiently small and $K$ is sufficiently large, we would produce a sample from $p(x)$ under some regularity conditions!

### Pitfalls of Naive Score-based Generative Modelling

There are pitfalls involved with this naive approach of sampling via Langevin dynamics. Firstly, in regions where few data points are available for training the model, the estimated score functions can be inaccurate. This is because the model is trained on the Fisher divergence which is an expectation over $p(x)$, i.e.

$$
\mathbb{E}_{p(x)} \Big[ \|  \nabla_x \log p(x) -  s_\theta(x)  \|^2_2 \Big] = \int p(x) \|  \nabla_x \log p(x) -  s_\theta(x)  \|^2_2 \, dx
$$

Because the squared $\ell_2$ difference is weighted by $p(x)$, the errors are largely ignored in low density regions where $p(x)$ is small and results in an inaccurate learned model. Since sampling via Langevin dynamics involves starting from a random location in the high-dimensional space which is likely to be in low density regions (it is unlikely to choose a point that corresponds to an actual sample), an inaccurate score-based model derails the sampling trajectory right from the beginning.

Secondly, the manifold hypothesis poses a problem. The manifold hypothesis postulates that real-world data often lies in a low-dimensional manifold embedded in a high-dimensional space and it has been empirically observed in many datasets. As such, points in the high-dimensional space outside of the low-dimensional manifold would have probability zero and an undefined $\nabla_x \log p(x)$. This causes the score function to be ill-defined. Moreover, in order to represent the objective using Equation 5, it would require that the support of $p(x)$ is over the whole space.

### **Score-based Generative Modeling with Multiple Noise Perturbations**

As it turns out, the above-mentioned pitfalls can be bypassed simply by perturbing data points with Gaussian noise. By injecting small amounts of Gaussian noise into the data, the noised data distribution would have full support and is no longer confined to a low-dimensional manifold. When the noise magnitude is sufficiently large, it would also populate low data density regions and thereby improve the accuracy of the score function. 

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed to perturb the data with a increasing sequence of isotropic Gaussian noise $\{\sigma_t\}_{t=1}^T$ to obtain a noise-perturbed distribution

$$
p_{\sigma_t}(x) = \int p(x) \mathcal{N} (x_t; x, \sigma_t^2I) \, dx
$$

Then, they estimate the score function of the noise-perturbed distribution by training a *Noise-Conditioned Score Network (NCSN)* $s_\theta (x, t)$ to jointly estimate the scores of all perturbed data at different noise levels. The objective then becomes

$$
\begin{aligned}
\hat{\theta} &= \argmin_{\theta} \sum_{t=1}^T \lambda (t)\mathbb{E}_{p_{\sigma_t}(x_t)} \Big[ \|  \nabla_{x_t} \log p_{\sigma_t} (x_t) -  s_\theta(x,t)  \|^2_2 \Big] \tag{6}
\end{aligned}
$$

where $\lambda(t)$ is a positive weighing function. After training $s_\theta (x, t)$, we can then produce samples by running Langevin dynamics for each $t = T, T-1, \dots, 1$ and this method is called **annealed Langevin dynamics** since the noise scale $\sigma_t$ decreases (anneals) with each time step.

Notice how this procedure is very similar to the one in DDPMs. In fact, their objectives can be reconciled with the same formulation and that is what we’re going to show. For a Gaussian data distribution $x \sim \mathcal{N}(\mu, \sigma^2 I)$, the derivative of the log density function is

$$
\begin{aligned}
\nabla_x \log p(x) &= \nabla_x \Big[ \log (\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\frac{(x_t - \mu)^2}{\sigma^2}}) \Big] \\
&= \nabla_x(-\frac{1}{2\sigma^2}(x-\mu)^2) \\
&= -\frac{x-\mu}{\sigma^2} \\
&= -\frac{\epsilon}{\sigma}
\end{aligned}
$$

where $\epsilon \sim \mathcal{N}(0, I)$. Recall from [earlier](https://www.notion.so/Notes-on-Diffusion-Models-143444a757834040a5dd5b6c935b62b8?pvs=21) in DDPM, the posterior of a noised $x_t$ given initial $x_0$ is $q(x_t | x_0) \sim \mathcal{N}(x_t; \sqrt{\bar{a}_t}x_0, (1-\bar{a}_t)\mathbf{I})$. As such, we can rewrite the gradient of the log density of each noised sample $x_t$ as:

$$
\begin{aligned}
\nabla_{x_t}\log p(x_t) 
&=\mathbb{E}_{q(x_0)}\Big[\nabla_{x_t}\log q(x_t | x_o) \Big] \\
&= \mathbb{E}_{q(x_0)}\Big[- \frac{\epsilon_0}{\sqrt{1-\bar{a}_t}}\Big] \\
&= \int-q(x)\cdot\frac{\epsilon_0}{\sqrt{1-\bar{a}_t}} \, dx \\
&= -\frac{\epsilon_0}{\sqrt{1-\bar{a}_t}}\int q(x) \, dx \\
&= -\frac{\epsilon_0}{\sqrt{1-\bar{a}_t}} \tag{6}
\end{aligned}
$$

where $\epsilon_0 \in \mathcal{N}(\epsilon; 0, I)$ refers to the source noise that produces $x_t$ from the original image $x_0$.

As it turns out, the gradient of the log density function in NCSN and the source noise in DDPM are off by only a constant factor that scales with time (i.e. $-\frac{1}{\sqrt{1-\bar{a}_t}}$)! Intuitively, the reason the score function in NCSN points in the opposite (negative) direction of the source noise in DDPM is that the score function indicates the direction in which the log density is maximized, whereas the source noise in DDPM corrupts the image away from its original state. Therefore, moving in the opposite direction of the source noise effectively "denoises" the image and increases the log density of the resulting sample. 

With that, we have shown that the $\epsilon$-prediction parameterization in DDPM simplifies the diffusion model’s variational bound to an objective that closely resembles denoising score matching in NCSN!

## Conditional Generation

As of now, we have only focused on modelling the sample distribution $p(x)$. However, it would be more practical if we could learn a conditional distribution $p(x|y)$ that allows us to generate images conditioned on conditional information $y$ which can be a class label, a text encoding in image-text generation, or even a low-resolution image to perform super-resolution on. Conditional generation forms the backbone of state-of-the-art image-to-text models such as [DALL-E 2](https://arxiv.org/abs/2204.06125v1) and [Imagen](https://arxiv.org/abs/2205.11487).

An intuitive way to incorporate conditioning information $y$ into the generative process would be to condition each transition step on it as such:

$$
p(x_{0:T} | y) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t, y)
$$

This would allow us to learn the model as before, by predicting the source noise $\hat{\epsilon}_\theta(x_t, t, y) \approx \epsilon_0$. However, this formulation may cause the model to ignore the conditioning information. As such, classifier guidance was introduced to control the amount of weight given to the conditioning information, at the cost of sample diversity. Classifier guidance provides a way to steer the sampling process in the direction of the image space that maximizes the probability of the denoised image belonging to the desired class.

### Classifier Guided Diffusion

The first way is to introduce guidance is by training a separate classifier model $p(y|x_t)$ on noisy  samples $x_t$. At every step $t$ of the denoising process, we can then use the gradients of the classifier, i.e. $\nabla_{x_T} \log p(y|x_t)$, to guide the denoising process towards an arbitrary input feature, which could be a class label, $y$. 

To understand why this works, we can make use of the Bayes rule:

$$
\begin{aligned}
\nabla_{x_t} \log p(x_t | y) &= \nabla_{x_t} \log\Big(\frac{p(x_t)p(y|x_t)}{p(y)}\Big) \\
&= \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t) - \nabla_{x_t} \log p(y) \\
&= \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{unconditional gradient}} + \underbrace{\nabla_{x_t} \log p(y|x_t)}_{\text{classifier gradient}} \tag{7}
\end{aligned}
$$

In order to have control over the degree to which the conditional model adheres to the conditioning information, [Dhariwal and Nichol (2021)](https://arxiv.org/abs/2105.05233) scales the conditioning term by a hyperparameter:

$$
\nabla_{x_t}\log p_\gamma(x_t |y) = \nabla_{x_t} \log p(x_t) + \gamma\nabla_{x_t} \log p(y|x_t) \tag{8}
$$

where $\gamma$ is called the guidance scale. $\gamma$ essentially controls how much our learned conditional model cares about the conditioning information. The greater $\gamma$ is, the more likely the generated samples reflect the conditioned information, albeit at the cost of sample diversity. Intuitively, the score function $\nabla_{x_t} \log p(x_t)$ points in the direction of the image space that maximises the likelihood of the resulting image and the classifier gradients $\nabla_{x_t} \log p(y|x_t)$ points in the direction of the image space that maximises the likelihood of the conditioning information. Combining these gradients together results in a step that balances between both objectives depending on $\gamma$.

![Image from [Dieleman (2023)](https://sander.ai/2023/08/28/geometry.html)](../assets/2025-08-16-diffusion-models/Untitled%204.png)

Equation 8 requires that we combine the output of a denoising score-based generative model and the gradients of the classifier. However, recall that denoising score matching has a similar interpretation as $\epsilon$-prediction parameterization in DDPM, i.e. $\log p(x_t) = -\frac{1}{\sqrt{1-\hat{a}_t}}\epsilon_\theta(x_t, t)$. Plugging this into Equation 8 we get:

$$
\begin{aligned}
\nabla_{x_t}\log p_\gamma(x_t |y) &= -\frac{1}{\sqrt{1-\hat{a}_t}}\epsilon_\theta(x_t, t) + \gamma \nabla_{x_t}\log p(y|x_t) \\
&= -\frac{1}{\sqrt{1-\hat{a}_t}} \Big[\epsilon_\theta(x_t, t) - \gamma \sqrt{1-\hat{a}_t} \nabla_{x_t}\log p(y|x_t)\Big]
\end{aligned}
$$

As such, our new classifier-guided DDPM would take the form:

$$
\epsilon_\theta (x_t;t,y) = \epsilon_\theta(x_t, t) - \gamma \sqrt{1-\hat{a}_t} \nabla_x\log p(y|x_t) \tag{9}
$$

There are a few things to clarify here:

1. Both the classifier and diffusion models are trained on noisy images independently.
2. The reason behind training the classifier on the noisy image samples instead of the original images is to allow it to be robust against the distortion introduced by the forward trajectory of the diffusion models. This is also what allows us to apply the gradients of the classifier for guidance on every step of the denoising process.
3. In the neural network paradigm, the gradient of the classifier is calculated by backpropagating through the layers of trained classifier from the output logits back to the input image

### Classifier-free Guidance

The drawback of the classifier guidance is that it requires the training of two separate models, the diffusion model and the classifier. Moreover, this classifier must be trained on noisy data so it is generally not possible to plug in a pre-trained classifier. As such, [Ho & Salimans (2021)](https://openreview.net/forum?id=qw8AKxfYbI) introduced *classifier-free diffusion guidance*, which allows us to jointly train a conditional and unconditional diffusion model using a singular neural network, without learning a separate classifier. This model, denoted $\epsilon_\theta(x_t;t,y)$, is trained on the labelled data $(x,y)$ but the conditioning information $y$ is discarded periodically to allow the model to generate images unconditionally, i.e. $\epsilon_\theta(x_t;t) = \epsilon_\theta(x_t;t,y =\varnothing)$. 

To train $\epsilon_\theta(x_t;t,y)$ without a classifier, we have to replace Equation 9 to not rely on its gradient. First, by switching the terms around in Equation 7 we get 

$$
\begin{aligned}
\nabla_{x_t}\log p(y|x_t) &= \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t) \\
&= -\frac{1}{\sqrt{1-\hat{a}_t}} \Big[\epsilon_\theta (x_t;t,y)  - \epsilon_\theta (x_t;t) \Big]
\end{aligned}
$$

Plugging this into Equation 9 we get:

$$
\begin{aligned}
\epsilon_\theta (x_t;t,y) &= \epsilon_\theta(x_t, t) - \gamma \sqrt{1-\hat{a}_t} \nabla_x\log p(y|x_t) \\
&= \epsilon_\theta(x_t, t) + \gamma \Big[\epsilon_\theta (x_t;t,y)  - \epsilon_\theta (x_t;t) \Big] \\
&= (\gamma+1)\epsilon_\theta (x_t;t,y) -  \gamma\epsilon_\theta (x_t;t)\\
\end{aligned}
$$

This new equation has no classifier gradient present. 

To implement this in a neural network, you can simply concatenate the conditioning information to the end of the flattened image vector and replace the conditioning information with fixed constant values (e.g. zeros) periodically. The neural network would then learn to treat those fixed constant values as the $\varnothing$ conditioning and perform unconditional generation when inputted with them.

## Summary

To wrap up, diffusion models represent a fascinating approach to data generation and denoising tasks in the realm of deep learning. At their core, these models leverage a process called diffusion, which involves progressively applying Gaussian noise to an original sample distribution. The diffusion model then aims to learn the parameters that maximize the Variational Lower Bound (VLB) during the denoising step. 

Notably, diffusion models often excel in terms of generalizability and training stability, steering clear of pitfalls like mode collapse commonly encountered by GANs. However, GANs are still considerably more computationally efficient compared to diffusion models. This is due to the fact that GANs can generate an image in a single forward pass but diffusion models rely on a long Markov chain of denoising steps.

Still, diffusion models have proved their incredible capabilities in generating realistic samples. It's highly likely that we'll witness the emergence of even more powerful diffusion-based generative models, pushing the boundaries of generative modeling to new heights in the years ahead!

## References

[1] Goodfellow, Ian & Pouget-Abadie, Jean & Mirza, Mehdi & Xu, Bing & Warde-Farley, David & Ozair, Sherjil & Courville, Aaron & Bengio, Y.. (2014). Generative Adversarial Networks. Advances in Neural Information Processing Systems. 3. 10.1145/3422622. 

[2] Dhariwal, Prafulla & Nichol, Alex. (2021). Diffusion Models Beat GANs on Image Synthesis. 

[3] Zhang, H., Xu, T., Li, H., Zhang, S., Wang, X., Huang, X., & Metaxas, D.N. (2016). StackGAN: Text to Photo-Realistic Image Synthesis with Stacked Generative Adversarial Networks. *2017 IEEE International Conference on Computer Vision (ICCV)*, 5908-5916.

[4] Xu, T., Zhang, P., Huang, Q., Zhang, H., Gan, Z., Huang, X., & He, X. (2017). AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks. *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1316-1324.

[5] Shoshan, A., Bhonker, N., Kviatkovsky, I., & Medioni, G.G. (2021). GAN-Control: Explicitly Controllable GANs. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, 14063-14073.

[6] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S.. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Proceedings of the 32nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 37:2256-2265

[7] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *ArXiv, abs/2006.11239*.

[8] Feller, W. (1949). On the Theory of Stochastic Processes, with Particular Reference to Applications.

[9] Weng, Lilian. (2021). What are diffusion models? Lil’Log. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[10] Luo, C. (2022). Understanding Diffusion Models: A Unified Perspective. *ArXiv, abs/2208.11970*.

[11] Karagiannakos, S., Adaloglou, N. (2022). Diffusion models: toward state-of-the-art image generation. [https://theaisummer.com/diffusion-models/](https://theaisummer.com/diffusion-models/)

[12] Song, Y., & Stefano Ermon. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *ArXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1907.05600

[13] Dhariwal, P., & Nichol, A. Q. (2021). Diffusion Models Beat GANs on Image Synthesis. *Neural Information Processing Systems*, *34*.

[14] Zeng, F. P., & Wang, O. (2023). Score-Based Diffusion Models. *fanpu.io*. [https://fanpu.io/blog/2023/score-based-diffusion-models/](https://fanpu.io/blog/2023/score-based-diffusion-models/)

[15] Hyvärinen, A. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. *Journal of Machine Learning Research*, *6*(24), 695–709. http://jmlr.org/papers/volume6/hyvarinen05a/old.pdf