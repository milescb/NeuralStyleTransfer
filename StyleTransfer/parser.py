import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--nEpochs', default=100, type=int)
parser.add_argument('--learning_rate', default=7.0, type=float)
parser.add_argument('--content_path', default='original_content/fernsehturm.jpeg', type=str)
parser.add_argument('--style_path', default='styles/starry_night.jpg', type=str)
parser.add_argument('--save_folder', default='test', type=str)