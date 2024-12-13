
# DAP 𝄇 : Develop pair-Authentication Protocol with DAP

This work corresponds to the following paper: [acm website](https://dl.acm.org/doi/abs/10.1145/3640471.3680449)

Project page: will be updated soon

```bibtex
@inproceedings{10.1145/3640471.3680449,
  author = {Maeda, Soshi and Okano, Masora and Nishigaki, Masakatsu and Ohki, Tetsushi},
  title = {DAP PLXENT#x1D107; : Develop pair-Authentication Protocol with DAP},
  year = {2024},
  isbn = {9798400705069},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3640471.3680449},
  doi = {10.1145/3640471.3680449},
  booktitle = {Adjunct Proceedings of the 26th International Conference on Mobile Human-Computer Interaction},
  articleno = {11},
  numpages = {6},
  keywords = {Cooperative actions, DAP, Pair authentication, Sensor, Wearables},
  location = {Melbourne, VIC, Australia},
  series = {MobileHCI '24 Adjunct}
}
```

## abstract

In today’s interconnected world, secure authentication is crucial for both high-security environments and everyday interactions. Traditional authentication methods like passwords and biometrics are designed for individual use, but new challenges emerge in interactive gaming, theme parks, and collaborative virtual reality (VR) where multiple participants must authenticate collectively. This study introduces a multi-person authentication that leverages cooperative actions to enhance security. By analyzing synchronized sensor data from cooperative actions, the system ensures the presence and consent of all participants, making impersonation difficult. We propose a pair authentication using inertial sensors during a complex handshake known as Dignity And Pride (DAP). Our research evaluates the accuracy of pair authentication, the impact of behavioral degradation over time, and resistance to attacks. Experiments with university students demonstrate high authentication accuracy and robustness against time degradation, though vulnerabilities to spoofing attacks were identified, suggesting areas for improvement in secure cooperative authentication.

## Setup

To execute this code, you will need two witmotion wt901c devices. Please check the addresses and create a new configuration file by referring to the configuration file that currently exists in `conf/devices/device`.

## Demo

In order to implement the demo code, you will need a trained model that has been certified for at least one pair.

If you don't want to go to the trouble of doing this, you can use `dap_auth_demo/weight/rf_None_and_None.pickle` as a pre-learned parameter. Please replace the `param_dict_path` item in `conf/model/rf.yaml` with `dap_auth_demo/weight/rf_None_and_None.pickle` and run it.

```shell
python src/demo.py
```

## Data sampling

Please overwrite the `devices` section of `dap_auth_demo/conf/data_sampling.yaml` with the configuration file name for your sensor that you created in the `Setup` chapter.

Please follow the instructions in the output text after entering the code below.

```shell
python src/data_sampling.py
```

## Train model

In order to learn, you need to sample the DAP operation in advance. Please refer to the `Data sampling` chapter.

Please overwrite the `devices` section of `dap_auth_demo/conf/train.yaml` with the configuration file name for your sensor that you created in the `Setup` chapter.

```shell
python src/train.py
```
