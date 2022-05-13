# DCIL_XPAG
Implementation of DCIL based on jax-based XPAG library.

# Install 

1. Create virtual environment dcil_env from environment.yalm,


conda env create --name dcil_env --file environment.yaml

2. Install xpag (requires Jax & Brax),

[Repo](https://github.com/perrin-isir/xpag)

3. Install maze environments 

```sh
https://github.com/AlexandreChenu/gmaze_dcil.git
```

and 

```sh
pip install -e .
```

4. Clone DCIL repo,


```sh
git clone https://github.com/AlexandreChenu/DCIL_XPAG.git
```

# Run Dubins Experiment

```sh
python test_DCIL_XPAG.py --demo_path ./demos/dubins_convert/1.demo --save_path /path/to/save/path
```

# Visual logs produced in /path/to/save/path

- trajs_it_- : training rollouts + skill-chaining evaluation
- value_skill_-_it_- : value for x-y position of skill starting state for different orientations 
- transitions_- : sampled training transitions + segment between true desired goal and relabelled desired goal

