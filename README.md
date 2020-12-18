[![codecov](https://codecov.io/gh/West-Coast-Differentiators/cs107-FinalProject/branch/master/graph/badge.svg?token=CVUJ0SI09S)](https://codecov.io/gh/West-Coast-Differentiators/cs107-FinalProject)
[![Build Status](https://travis-ci.com/West-Coast-Differentiators/cs107-FinalProject.svg?token=LcEGi8DXzVyEeNU9JqUx&branch=master)](https://travis-ci.com/West-Coast-Differentiators/cs107-FinalProject)

# cs107-FinalProject

## Group 14
* Anita Mahinpei
* Yingchen Liu
* Erik Adames
* Lekshmi Santhosh


# Project description

Our package contains several optimization algorithms that are ubiquitous in machine learning. In the context of the WestCoastAD package, an optimization problem refers to the minimization of a function. Below are the optimization that are currently included in this package:
  * Gradient Descent
  * Momentum Gradient Descent
  * AdaGrad
  * RMSprop
  * Adam
  * BFGS

All the optimization methods mentioned above require derivative computations. For this library, we have used Automatic Differentiation as it is an efficient way of computing these derivatives which can be used with various complex functions.

[Introductory video](https://youtu.be/qErU1uYU85E)

# Broader Impact and Inclusivity Statement

### Broader Impact

Optimizers have a wide range of applications across various industries. Some of the use cases include 
cost/revenue optimization, optimal labor staffing at factories, automated location navigation, supply chain optimization, work allocation in crowd-sourcing markets.
`WestCoastAD` enables automatic optimization of real word problems, modeled as objective functions for optimal decision making.
There is more work to be done as elaborated in Futures Section but what we have currently is a step in the right direction.

Based on the usage context, optimization errors and lack of understanding of how optimizers work can affect societies in varying degrees.
For instance, inaccurate optimizations can result in delays/increased inventory costs in supply chain management. Knowledge about how an optimizer does optimization can have positive and negative impacts.
Per study elaborated in [6], lack of transparency in algorithmic management affected the morale of employees.
On the contrary, there were scenarios where knowledge of how the algorithm optimized the objective encouraged offenders to game the system.

It is important to design optimizers in `WestCoastAD` that converge faster with high accuracy. There is the potential misuse of showcasing these methods as methods that can find optimal results but in reality they are very sensitive to parameter settings and unless the optimization function is convex, there is no guarantee that the algorithm will be close to the true global optimum.

### Software Inclusivity

We plan to release this software to the open source community for contribution.
The name `WestCoastAD` was something we came up with for the purposes of this class (since the core contributors were all on the Pacific time zone).
Prior to open sourcing, we will rename our package to make it location neutral.
We will also rename repo `master` branch to `main` per [Github guidelines](https://github.com/github/renaming).
We will hold the contributors and peer reviews accountable to a code review standard.
Google has published it's [engineering code review](https://github.com/google/eng-practices/blob/master/review/index.md) practice which we can adopt.
We welcome all members of the software community to contribute to this package. 
In addition to functionality extension, there are opportunities to localize the software to make it accessible for contributors across all regions.
We would translate the documentation to multiple languages.
