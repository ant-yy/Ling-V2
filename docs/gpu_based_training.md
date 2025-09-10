# Environment Preparation
## 1. Clone Ling-V2
## 2. Prepare [Docker Environment](../training/megatron_based_training.dockerfile)
## 3. Apply Patch
```bash
cd Ling-V2
# apply megatron patch
bash training/megatron/apply_patch.sh
# apply te patch
bash training/te/apply_te_patch.sh
```

# Continual Pretraining
## Prepare Data
### 1. Download [oscar-en-10k](https://huggingface.co/datasets/stas/oscar-en-10k) dataset
```bash
python examples/pretrain/download_example_data.py
```
Dataset will be saved in `oscar-en-10k.jsonl`

### 2. Run megatron preprocessing
```bash
bash examples/pretrain/preprocess_data.sh
```
Data will be saved in  `processed_data_text_document.bin` and `processed_data_text_document.idx`

For more details, please refer to the [megatron documentation](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.13.0?tab=readme-ov-file#data-preprocessing).


## Run Pretraining
Set `MODEL_PATH` in [run_pretrain_8k.sh](../examples/pretrain/run_pretrain_8k.sh)

Run pretraining:
```bash
bash examples/pretrain/run_pretrain_8k.sh
```
Example log:
```log
 [2025-09-05 16:08:14] iteration        1/    2000 | consumed samples:          128 | elapsed time per iteration (ms): 79489.9 | throughput per GPU (TFLOP/s/GPU): 18.0 | learning rate: 2.999998E-05 | global batch size:   128 | lm loss: 2.605958E+00 | z_loss: 1.091659E+01 | mtp_1 loss: 2.828435E+00 | loss scale: 1.0 | grad norm: 0.258 | number of skipped iterations:   0 | number of nan iterations:   0 |
[Rank 0] (after 1 iterations) memory (MB) | allocated: 26222.7236328125 | max allocated: 63359.77001953125 | reserved: 66802.0 | max reserved: 66802.0
 [2025-09-05 16:08:25] iteration        2/    2000 | consumed samples:          256 | elapsed time per iteration (ms): 11182.0 | throughput per GPU (TFLOP/s/GPU): 128.2 | learning rate: 2.999993E-05 | global batch size:   128 | lm loss: 2.688647E+00 | z_loss: 1.019938E+01 | mtp_1 loss: 2.843235E+00 | loss scale: 1.0 | grad norm: 7.737 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:08:36] iteration        3/    2000 | consumed samples:          384 | elapsed time per iteration (ms): 10883.5 | throughput per GPU (TFLOP/s/GPU): 131.7 | learning rate: 2.999985E-05 | global batch size:   128 | lm loss: 2.653257E+00 | z_loss: 1.102731E+01 | mtp_1 loss: 2.874039E+00 | loss scale: 1.0 | grad norm: 0.306 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:08:47] iteration        4/    2000 | consumed samples:          512 | elapsed time per iteration (ms): 10840.4 | throughput per GPU (TFLOP/s/GPU): 132.2 | learning rate: 2.999973E-05 | global batch size:   128 | lm loss: 2.656384E+00 | z_loss: 1.094066E+01 | mtp_1 loss: 2.883725E+00 | loss scale: 1.0 | grad norm: 0.178 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:08:58] iteration        5/    2000 | consumed samples:          640 | elapsed time per iteration (ms): 10765.6 | throughput per GPU (TFLOP/s/GPU): 133.1 | learning rate: 2.999958E-05 | global batch size:   128 | lm loss: 2.571362E+00 | z_loss: 1.115748E+01 | mtp_1 loss: 2.793797E+00 | loss scale: 1.0 | grad norm: 0.160 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:09:08] iteration        6/    2000 | consumed samples:          768 | elapsed time per iteration (ms): 10827.5 | throughput per GPU (TFLOP/s/GPU): 132.4 | learning rate: 2.999940E-05 | global batch size:   128 | lm loss: 2.561039E+00 | z_loss: 1.079292E+01 | mtp_1 loss: 2.779968E+00 | loss scale: 1.0 | grad norm: 0.141 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:09:19] iteration        7/    2000 | consumed samples:          896 | elapsed time per iteration (ms): 10767.4 | throughput per GPU (TFLOP/s/GPU): 133.1 | learning rate: 2.999918E-05 | global batch size:   128 | lm loss: 2.537825E+00 | z_loss: 1.044740E+01 | mtp_1 loss: 2.759917E+00 | loss scale: 1.0 | grad norm: 0.156 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:09:30] iteration        8/    2000 | consumed samples:         1024 | elapsed time per iteration (ms): 10758.5 | throughput per GPU (TFLOP/s/GPU): 133.2 | learning rate: 2.999893E-05 | global batch size:   128 | lm loss: 2.645642E+00 | z_loss: 1.094773E+01 | mtp_1 loss: 2.867552E+00 | loss scale: 1.0 | grad norm: 0.158 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:09:41] iteration        9/    2000 | consumed samples:         1152 | elapsed time per iteration (ms): 10727.4 | throughput per GPU (TFLOP/s/GPU): 133.6 | learning rate: 2.999865E-05 | global batch size:   128 | lm loss: 2.549589E+00 | z_loss: 1.108937E+01 | mtp_1 loss: 2.770095E+00 | loss scale: 1.0 | grad norm: 0.154 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-09-05 16:09:51] iteration       10/    2000 | consumed samples:         1280 | elapsed time per iteration (ms): 10720.6 | throughput per GPU (TFLOP/s/GPU): 133.7 | learning rate: 2.999833E-05 | global batch size:   128 | lm loss: 2.557783E+00 | z_loss: 1.095377E+01 | mtp_1 loss: 2.779469E+00 | loss scale: 1.0 | grad norm: 0.144 | number of skipped iterations:   0 | number of nan iterations:   0 |
```