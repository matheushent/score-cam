# score-cam
Repository containing code to run Score-CAM algorithm available on https://arxiv.org/pdf/1910.01279v1.pdf.

## Pre-requisites

Make sure you have installed python 3.6+ and setup [GPU configuration](https://www.tensorflow.org/install/gpu) in case you have GPU available.

At the root run:

```
$ virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Running

To run a simple example just enter ```python example.py``` on terminal. Output image will be saved at the root.

![alt text](./score_cam.png)

## Upcoming

* Visualize more than one image per run
* Implement on [tf-explain](https://github.com/sicara/tf-explain)

## ATTENTION

By now, something is going wrong. Feel free to open an issue and/or submit a pull request.