# Segmentation-Based Deep-Learning Approach for Surface-Defect Detection
## Try to make this as a productive project(on going)
## Implement this paper by Pytorch
[SDASDD](https://link.springer.com/article/10.1007/s10845-019-01476-x)

## Network Arch
![network arch](paper/imgs/arch.jpg)

## usage
### prepare your KolektorSDD dataset
    1. under KolktorSDD dir, create two txt files('train.txt','val.txt')
    2. write the image filename and label filename like below:
    ```
    kos01/Part5.jpg kos01/Part5_label.bmp
    kos02/Part6.jpg kos02/Part6_label.bmp
    kos03/Part2.jpg kos03/Part2_label.bmp
    kos04/Part3.jpg kos04/Part3_label.bmp
    ...
    ```
### start training
    1. modify 'train.py'
    DATAROOT
    GLOBALEPOCH
    INPUTHW
    2. python train.py
## pre-trained model(google drive)
   1. [pre-trained segment net](https://drive.google.com/file/d/114hJy6id0KeqowsYOfoLPkt-jcRDxic5/view?usp=sharing) 
   2. [pre-trained decision net]()
## TODO
- [x] Forward finished
- [x] Segmentation Net Training & Validate functions
- [ ] Decision Net Training & Validate functions
- [x] resume segmentation net training script
- [ ] resume decision net training script
- [ ] Tensorboard record
- [ ] Model fuse
- [ ] ONNX format
- [ ] Windows deployment
- [ ] Linux deployment
## Support Me
If you want to speedup my progress, plsease support me.  
<a href="https://www.buymeacoffee.com/winafoxq"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=winafoxq&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" /></a>
