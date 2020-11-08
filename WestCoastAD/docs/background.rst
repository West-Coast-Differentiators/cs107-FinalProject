.. _background:

============
Background
============

Automatic differentiation assumes that we are working with a differentiable function composed of a finite number of elementary functions and operations with known symbolic derivatives. The table below shows some examples of elementary functions and their respective derivatives:

| Elementary Function    | Derivative              |
| :--------------------: | :----------------------:|
| $x^3$                  | $3 x^{2}$               | 
| $e^x$                  | $e^x$                   |
| $\sin(x)$              | $\cos(x)$               |
| $\ln(x)$               | $\frac{1}{x}$           |

Given a list of elementary functions and their corresponding derivatives, the automatic differentiation process involves the evaluations of the derivatives of complex compositions of these elementary functions through repeated applications of the chain rule:

$ \frac{\partial}{\partial x}\left[f_1 \circ (f_2 \circ \ldots (f_{n-1} \circ f_n)) \right] = 
\frac{\partial f_1}{\partial f_2} \frac{\partial f_2}{\partial f_3} \ldots \frac{\partial f_{n-1}}{\partial f_n}\frac{\partial f_n}{\partial x}$

This process can be applied to the evaluation of partial derivatives as well thus allowing for computing derivatives of multivariate and vector-valued functions.

## Computational Graph

The forward mode automatic differentiation process described above can be visualized in a computational graph, a directed graph with each node corresponding to the result of an elementary operation.

For example, consider the simple problem of evaluating the following function and its derivative at $x=2$:
$$
f(x) = x^3 +2x^2
$$
The evaluation of this function can be represented by the evaluation trace and computational graph below where the numbers in orange are the function values and the numbers in blue are the derivatives evaluated after applying each elementary operation:

<img src="https://raw.githubusercontent.com/anita76/playground/master/src/ex_comp_graph.png" width="75%" />

| Trace       | Elementary Function     | Value   | Derivative                | Derivative Value      |
| :---------: | :----------------------:| :------:| :------------------------:| :--------------------:|
| $v_1$       | $x$                     | $2$     | $\dot{x}$                 | $1$                   |
| $v_2$       | $v_1^2$                 | $4$     | $2v_1 \dot{v}_1$          | $4$                   |
| $v_3$       | $v_1^3$                 | $8$     | $3v_1^2 \dot{v}_1$        | $12$                  |
| $v_4$       | $2v_2$                  | $8$     | $2 \dot{v}_2$             | $8$                   |
| $v_5$       | $v_3 + v_4$             | $16$    | $ \dot{v}_3 + \dot{v}_4$  | $20$                  |
∂fn​​

This process can be applied to the evaluation of partial derivatives as well thus allowing for computing derivatives of multivariate and vector-valued functions.