## Grad-CAM Demo
This repository contains a Python script that demonstrates how to use Grad-CAM to generate a heatmap that highlights the regions of an image that contributed to a convolutional neural network's (CNN) prediction.

### Dependencies
The following Python packages are required to run the script:

- torch
- torchvision
- opencv-python-headless 
- numpy

You can install these packages using the following command:

```pip install torch torchvision opencv-python-headless numpy```


where **path/to/image** is the path to the input image file and **path/to/model** is the path to the pre-trained CNN model file.

The script will generate a Grad-CAM heatmap for the input image and save it to a file named **grad_cam.png** in the same directory as the input image.

## Example
Here's an example of how to use the script:

```
chmod +x run_inference.sh
./run_inference.sh
```

## References

For more information on Grad-CAM, see the following papers:

- Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." International Conference on Computer Vision (ICCV), 2017.
- Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2020). Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks. IEEE Transactions on Image Processing, 29, 4763–4778. https://doi.org/10.1109/TIP.2020.2974875
License