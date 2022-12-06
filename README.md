# Neural-Style Transfer

Here, we follow the implementation given in [this Tensorflow example](https://blog.tensorflow.org/2018/08/neural-style-transfer-creating-art-with-deep-learning.html). We had to edit this code in order to run properly following, to some extent, the implementation given [here](https://github.com/ravising-h/Neural-Style-Transfer). The main idea is to use a convolution neural network (CNN) to map the style from one picture onto the other. This is made possible with the pre-trained network `VGG19` in `Tensorflow`. This network takes as an input the pixels and RBG colors for each pixel of both an original image and a style image, thus transfering the style onto the original photo. Note that I did not write the majority of the functions in `StyleTransfer.py` but rather corrected several bugs here and collected the code into an easily useable format. See below for the methods implemented for running the code. 

## Subject: the City of Berlin, Styles: various from German 19th centrury art

This project is done specifically for an art class I took in Berlin while studying abroad. In the class, we discussed art styles and movements from the 19th centry in Germany. My final project was focused on using these art styles and applying them to pictures I took around Berlin. 

### Subject Photos

I chose as my subject matter the following photos of iconic Berlin scenes:

Haus das Lehers mit Fernsehturm in Hintergrund

![](https://github.com/lvb5/NeuralStyleTransfer/blob/master/original_content/fernsehturm.jpeg)

## Running the code

After downloading required packages such as `tensorflow`, the code should run out of the box via

```
python3 perform_transfer.py
```

Possible command-line arguments are

```
python3 perform_transfer.py --nEpochs=100 --learning_rate=10.0 --content_path=<path_to_content> --style_path=<path_to_styles> --save_folder=<path_to_save_folder>
```

## Preparing Photos

Both style and content photos _must_ be the same dimension in order to run. In order to not consume too much memory / time in training, photos should be 500x500 pixels or smaller. After preparing photos as such, put them in appropriate folders and run the code!