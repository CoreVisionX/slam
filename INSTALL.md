## Jetson
1. Install librealsense by following these [instructions](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md).
2. Activate a Python 3.11 virtual environment
3. pip install (github releases wheel link)
4. Copy the config file you need from [here](config files link) (e.g. vio_d435i.yaml for running vio on the d435i) to your project's directory
5. Try copying the [vio_example_realsense.py](vio_example_realsense file link) example and running it!

## Pi
TODO

## Macos
1. Activate a Python 3.11 virtual environment
2. pip install (github releases wheel link)
3. Copy the config file you need from [here](config files link) (e.g. vio_euroc.yaml for running vio on the euroc benchmark) to your project's directory
4. download and extract any test sequences you'd like to use, e.g. any of the euroc sequences [here](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
5. Try copying the [vio_example_euroc.py](vio_example_euroc file link) example and running it!
