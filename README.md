# Adversarial Robust Decision Transformer

* You will need to clone the forked repository `stochastic_offline_envs` from [here](https://github.com/xiaohangt/stochastic_offline_envs).
* You will also need to clone the forked repository `arrl` from [here](https://github.com/afonsosamarques/arrl).

To install required packages, simply run
```bash
pip install -r ardt/requirements.txt
```

Currently this repo is built on top of the `gym` package. Due to version incompatibility with numpy, a simple and quick manual code change needs to be done to the package code. Access `.../gym/utils/passive_env_checker.py` and replace all instances of `np.bool8` with `np.bool_`. This file will sit inside the gym package, wherever in your local filesystem you store your Python environments and packages (e.g. `Users/username/.virtualenvs/package_name/lib/python3.12/site-packages/gym`).


### Instructions
* Model and training configurations for the return transformations models (ARDT, ESPER) sit under the `configs` directorys.
* Model and training configurations for the protagonist models (DT, ADT, BC) are passed on through the arguments fed into `main.py`.
* Pre-set run scripts with the parameters used in the results reported in our paper can be found under the `run-scripts` directory.


### References
We start from the ESPER codebase to develop our solution:
* Paster, Keiran, Sheila McIlraith, and Jimmy Ba. "You can’t count on luck: Why decision transformers and rvs fail in stochastic environments." In Advances in Neural Information Processing Systems 35 (2022): 38966-38979.

For our online data collection policy:
* Tessler, Chen, Yonathan Efroni, and Shie Mannor. "Action robust reinforcement learning and applications in continuous control." In International Conference on Machine Learning, pp. 6215-6224. PMLR, 2019.
