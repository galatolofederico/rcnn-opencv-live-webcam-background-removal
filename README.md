# rcnn-opencv-live-webcam-background-removal


![](README.md.d/lena.jpg)  |  ![](README.md.d/lena_bgrm.jpg) 
:-------------------------:|:-------------------------:
![](README.md.d/messi5.jpg)  |  ![](README.md.d/messi5_bgrm.jpg)



This project uses a pretrained Mask R-CNN from Detectron2 to compute the segmentation mask of a person in the shot and follows it using the OpenCV MOSSE tracker. The person bounding box and segmentation mask are updated accordingly with the Mask R-CNN throughput.


## Installation

To install the project you need to clone it

```
git clone https://github.com/galatolofederico/rcnn-opencv-live-webcam-background-removal.git && cd rcnn-opencv-live-webcam-background-removal
```

And install the dependencies in a conda virtualenv

```
conda env create -f environment.yml
```

## Usage

To run the project firstly you need to activate the conda virtualenv

```
conda activate webcam-background-removal
```

And then you can see the output with

```
python run.py --window
```

You can output the raw video to the stdout with

```
python run.py --raw
```


## Examples

Outputting the video to VLC

```
python run.py --raw | vlc --demux=rawvideo --rawvid-fps=30 --rawvid-width=640 --rawvid-height=480 --rawvid-chroma=RV24 - --sout '#display'
```

Streaming the video via UDP

```
python run.py --raw --bgr | ffmpeg -f rawvideo -pix_fmt rgb24 -s 640x480 -r 30 -i - -an -f mpegts udp://0.0.0.0:5555
```

Forwarding the video to a virtual webcam with v4l2loopback

```
python run.py --raw --bgr | ffmpeg -f rawvideo -pix_fmt bgr24 -s 640x480 -r 30 -i - -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```

## Known caveats 

If you have a gcc version higher than 8 you need to set `CC` and `CCX` before installing Detectron2

```
CC=gcc-8 CCX=g++-8 pip install git+https://github.com/facebookresearch/detectron2.git
```

## License

This code is released under the GPLv3, and it is free and open source as all the code should be. Feel free to do [whatever you want](https://choosealicense.com/licenses/gpl-3.0/) with it :D.

Issues or PRs are very welcome.