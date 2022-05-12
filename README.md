# DCIL_XPAG
Implementation of DCIL based on jax-based XPAG library.

# Install 

1. Create virtual environment dcil_env from environment.yalm,
conda env create --name dcil_env --file environment.yaml

2. Install xpag (requires Jax & Brax),
[Repo](https://github.com/perrin-isir/xpag)

3. Clone DCIL repo,
```sh
git clone https://github.com/AlexandreChenu/DCIL_XPAG.git
```

# Launch Dubins Experiment

```sh
python test_DCIL_XPAG.py --demo_path ./demos/dubins_convert/1.demo --save_path /path/to/save/path
```

