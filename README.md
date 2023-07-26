# Python package about interferometry using two-mode quantum states

## This package provides functions that return various physical quantities related to interferometry experiments using twin Fock states $|n,n\rangle$ or two-mode squeezed vacuum states

### Functions contained in this package are derived in [this article](link)
* DOI: *paper published soon*

## Quick context explanation

The scheme of the experiment that we aer considering is represented with the figure below:
![scheme_experiment](images/schematic_exp.png)
* **twin Fock** or **two-mode squeezed** states are used as *input* in a Mach-Zehnder interferometer ;
* experimentalists are interested in the measurement of the phase difference $\phi$ between the two arms ;
* the detection suffers from a **non-unit quantum efficiency** $\eta$ ;
* detailed explanation about the derivation of the formulae are given in the [supplemental material](link) of the article ;

## Examples

The following figure shows the **ratio** between the **phase sensitivity** achieved with those two quantum states **and the shot-noise**. The quantum efficiency is set to $\eta = 0.95$. It shows in particular that interferometry [below the standard quantum limit (SQL)](https://arxiv.org/abs/1405.7703) can be obtained in specific ranges of phase differences $\phi$.

![tf_vs_tms](images/FIG_tf_vs_tms/qsipy_tf_vs_tms.png)
The source code generating this figure [can be found here](images/FIG_tf_vs_tms/qsipy_tf_vs_tms.py).