# AKF-SR: Adaptive Kalman filtering-based successor representation

Implementation of the AKF-SR algorithm presented in the paper [AKF-SR: Adaptive Kalman filtering-based successor representation](https://arxiv.org/pdf/2204.00049.pdf) to play tag in OpenAI's [multi-agent particle environment](https://github.com/openai/multiagent-particle-envs).

## Code Structure
```
AKF-SR Codebase
│   AKFSR.py - AKF-SR model
│   AKFSR_tag.py - Run the AKFSR model
|   make_env.py - Create the environment by importing a multiagent environment as an OpenAI Gym-like object
│   AKFSR_initial.py - Initialize AKF-SR's parameters
│   general_utilities.py - Utilities for  directory creation, JSON data dumping, and managing time-series data in AKFSR_tag.py 
│   simple_tag_utilities.py- Utilities for the environment


```

## Paper Citation:
```
@article{malekzadeh2022akf,
  title={AKF-SR: Adaptive Kalman filtering-based successor representation},
  author={Malekzadeh, Parvin and Salimibeni, Mohammad and Hou, Ming and Mohammadi, Arash and Plataniotis, Konstantinos N},
  journal={Neurocomputing},
  volume={467},
  pages={476--490},
  year={2022},
  publisher={Elsevier}
}
```

## Credits

* The implementation of the multiagent environment is from [OpenAI](https://github.com/openai/multiagent-particle-envs).
