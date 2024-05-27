# RLNav

## Training

To start training, run one of the following commands:

```bash
python3 train_SARL.py
```

or

```bash
python3 train_RGL.py
```

Results, models, and tensorboard logging will be saved in the `logs` folder. The folder name will follow the format: `{model_name}_{version_number}`. Each model name will follow the format: `{model_name}_{version_number}_{training_step}`.

## Testing

In `test_SARL.py` and `test_RGL.py`, update the default `argparse` arguments for `model_name`, `version_num`, and `trained_steps`.

To test a model, run:

```bash
python3 test_SARL.py
```

or

```bash
python3 test_RGL.py
```