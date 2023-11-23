# Text2Speech project

## Report

The introduction, model architecture, implementation details, experiments, and results are presented in the [wandb report](https://wandb.ai/practice-cifar/tts_project/reports/Text2Speech-Report--Vmlldzo2MDUyMTMy).

## Installation guide

To get started install the requirements
```shell
pip install -r ./requirements.txt
```

Then download train data
```shell
sudo apt install axel
bash loader.sh
```

## Model training

This project implements [FastSpeech2](https://arxiv.org/abs/2006.04558) architecture for Text2Speech task.

To train model from scratch run
```shell
python3 train.py -c tts/configs/train.json
```

For fine-tuning pretrained model from checkpoint, `--resume` parameter is applied.
For example, continuing training model with `train.json` config organized as follows
```shell
python3 train.py -c tts/configs/train.json -r saved/models/1_initial/<run_id>/model_best.pth
```

## Inference stage

Before applying model pretrained checkpoint is loaded by python code
```python3
import gdown
gdown.download("https://drive.google.com/uc?id=1NvQ-TpAdKKEsNIdkdwWHxITzAfEXdMG6", "default_test_model/checkpoint.pth")
```

Model evaluation is executed by command
```shell
python3 test.py \
   -i default_test_model/text.txt \
   -r default_test_model/checkpoint.pth \
   -w waveglow/pretrained_model/waveglow_256channels.pt \
   -o output \
   -l False
```

- `-i` (`--input-text`) provide the path to input `.txt` file with texts. The file is readed by rows.
- `-r` (`--resume`) provide the path to model checkpoint. Note that config file is expected to be in the same dir with name `config.json`.
- `-w` (`--waveglow-path`) provide the path to pretrained `WaveGlow` model.
- `-o` (`--output`) specify output directory path, where `.wav` files will be saved.
- `-l` (`--log-wandb`) determine log results to wandb project or not. If `True`, authorization in command line is needed. Name of project can be changed in the config file.

Running with default parameters
```shell
python3 test.py
```

The model supports applying different coefficients of audio length, pitch, and energy.
For the inference stage they are taken from `{0.8, 1.0, 1.2}`, results will be written as follows
```python3
f"{row_number_in_txt_file_starting_from_1}-{length_level}-{pitch_level}-{energy_level}.wav"
```

Examples of model evaluation are also presented in [report](https://wandb.ai/practice-cifar/tts_project/reports/Text2Speech-Report--Vmlldzo2MDUyMTMy).

*Note:* `WaveGlow` model from `glow.py` supports only `GPU` by default. To run the code on `CPU`, remove all cuda tensors from this file with the command
```shell
sed -i `s/torch.cuda/torch/g` glow.py
```

Going back to `CUDA` version
```shell
cp FastSpeech/glow.py .
```

## Credits

The code of model is based on a [notebook](https://github.com/XuMuK1/dla2023/blob/2023/week07/seminar07.ipynb).
