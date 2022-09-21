# DCIL_XPAG
Implementation of DCIL-II based on jax-based XPAG library. 

# Install 

1. Clone DCIL repo,

```sh
git clone https://github.com/AlexandreChenu/DCIL_XPAG.git
```

2. Create virtual environment dcil_env from environment.ylm,


If your want to use Mujoco environments: 
```sh
cd DCIL_XPAG
conda env create --name dcil_env --file environment.yml
```

If you want to use PyBullet environments:

```sh
cd DCIL_XPAG
conda env create --name dcil_env_pybullet --file environment_pybullet.yml
```


3. Clone + install XPAG (+ Jax),

```sh
git clone https://github.com/perrin-isir/xpag.git
cd xpag
git checkout 9ef7dd74b74fc71cee83c6a476adfebe4b977814
pip install -e .
```

Check this [Repo](https://github.com/perrin-isir/xpag) for instructions.

4. Install physical simulators, 

[Mujoco](https://github.com/openai/mujoco-py)

[PyBullet](https://pypi.org/project/pybullet/)

[Brax](https://github.com/google/brax)


4. Clone + install maze or humanoid environments 

```sh
git clone https://github.com/AlexandreChenu/gmaze_dcil.git
```
OR

```sh
git clone https://github.com/AlexandreChenu/ghumanoid_dcil.git
```

and 

```sh
cd <env_directory>
pip install -e .
```

# Run Dubins Experiment

```sh
python test_DCIL_variant_XPAG_v4.py --demo_path ./demos/dubins_convert/1.demo --save_path /path/to/save/path
```

# Run Humanoid Experiment (Mujoco version)

```sh
python test_DCIL_variant_XPAG_humanoid_v4.py --demo_path ./demos/humanoid_convert/1.demo --save_path <path_to_results_directory> --eps_state 0.5  --value_clipping 1
```

(should work and learn sequential goal reaching with less than 1m training steps)

# Run Humanoid Experiment (PyBullet version) 

```sh
python test_DCIL_variant_XPAG_humanoid_walk_PB_v4.py --demo_path <path_to_this_directory>/demos/humanoid_PB_walk/ --save_path <path_to_results_directory> --eps_state 0.2  --value_clipping 1
```

(not working at the moment. Code running but no skill learning) 

> **_NOTE:_**  PyBullet installation requires python==3.8  

# Visual logs produced in /path/to/save/path

- trajs_it_- : training rollouts + skill-chaining evaluation + success goals sets 
- value_skill_-_it_- : value for x-y position of skill starting state for different orientations 
- transitions_- : sampled training transitions + segment between true desired goal and relabelled desired goal

