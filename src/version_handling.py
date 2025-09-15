#This file is intended for handling any package incompatibilities without resorting to full microservice containerisation
#for two reasons. 1) Runtime is longer for containerised solutions as the images are large if using IO operations to
# write the image, or segmentation outputs etc. Unless 2) we use a shared memory for the microservice, we can't bypass
# this bottleneck and it is unclear what the downtime would be for this.
import monai
monai_version = monai.__version__