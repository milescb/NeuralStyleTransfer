import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from StyleTransfer.StyleTransfer import run_style_transfer

from StyleTransfer.parser import parser
args = parser.parse_args()

#%% -----------------------------------------------------------------------
# Show images to train on
content_path = args.content_path
style_path = args.style_path

plt.figure(figsize=(10,5))

content = image.load_img(content_path)
content = image.img_to_array(content)/255.0

style = image.load_img(style_path)
style = image.img_to_array(style)/255.0

plt.subplot(1, 2, 1)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.imshow(content)

plt.subplot(1, 2, 2)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.imshow(style)

if not os.path.isdir(os.path.join('finished_product', args.save_folder)):
    os.mkdir(os.path.join('finished_product', args.save_folder))

plt.savefig(os.path.join('finished_product', args.save_folder, "initial.jpeg"), 
    bbox_inches='tight')

#%% ------------------------------------------------------------------------
# Perform actual learning + style transfer
best, best_loss, ani = run_style_transfer(content_path, style_path, 
                                        learning_rate=args.learning_rate,
                                        num_iterations=args.nEpochs,
                                        display_num=args.display_num)

print(best_loss)

plt.figure()
plt.imshow(best[0,:]/255.)
plt.tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)
plt.savefig(os.path.join('finished_product', args.save_folder, "final.jpeg"), 
    bbox_inches='tight')

writer = animation.PillowWriter(fps=10) 
ani.save(os.path.join('finished_product', args.save_folder, "final.gif"), writer=writer)