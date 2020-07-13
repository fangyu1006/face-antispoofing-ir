# face-antispoofing-ir

This project is for face-antispoofing based on ir cameras.

## Dataset
* Prepare dataset (training, validation and test): real faces, spoof faces on
  printed paper and screens.

## Train
* Choose a backbone.
```Shell
python train.py
```

## Converter
* keras h5 file to pb file

```Shell
cd tools
python h5Topb.py
```

* keras h5 file to tflite file

```Shell
python h5Totflite.py
```

## Test
```Shell
cd test
python test_irspoofing_keras.py
```
