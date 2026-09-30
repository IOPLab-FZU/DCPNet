# DPSM-Net
Directional Prompt-Guided State-Space Modeling for Remote Sensing Image Change Captioning
## Dataset
The data structure of LEVIR-CC is organized as follows:

```text
/root/Data/LEVIR_CC/
├── LevirCCcaptions.json
└── images/
    ├── train/
    │   ├── A/
    │   └── B/
    ├── val/
    │   ├── A/
    │   └── B/
    └── test/
        ├── A/
        └── B/
```

## Train

```
python train.py
```

## Test

```
python test.py --checkpoint /path/to/CHECKPOINT.pth
```

### Acknowledgement

The authors would like to thank the contributors to the [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC/tree/main), [Dubai-CC](https://disi.unitn.it/~melgani/datasets.html) and [WHU-CDC](https://huggingface.co/datasets/hygge10111/RS-CDC) datasets.
