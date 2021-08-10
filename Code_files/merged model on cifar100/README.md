**train accuracy:  tensor(0.4817) <br />
test accuracy:  tensor(0.4348) <br />
valid accuracy:  tensor(0.4186) <br />**
 <br />
 <br />
**Number of Trainable parameters :  185124**
 <br />
 <br />
SqueezeNet( <br />
  (features): Sequential( <br />
    (0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2)) <br />
    (1): ReLU(inplace=True) <br />
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True) <br />
    (3): Fire( <br />
      (squeeze): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), groups=16) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (4): Fire( <br />
      (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), groups=16) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (5): Fire( <br />
      (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), groups=32) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True) <br />
    (7): Fire( <br />
      (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), groups=32) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (8): Fire( <br />
      (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1), groups=48) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (9): Fire(
      (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1), groups=48) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (10): Fire( <br />
      (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), groups=64) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
    (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True) <br />
    (12): Fire( <br />
      (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)) <br />
      (squeeze_activation): ReLU(inplace=True) <br />
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), groups=64) <br />
      (expand1x1_activation): ReLU(inplace=True) <br />
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64) <br />
      (expand3x3_activation): ReLU(inplace=True) <br />
    ) <br />
  ) <br />
  (classifier): Sequential( <br />
    (0): Dropout(p=0.5, inplace=False) <br />
    (1): Conv2d(512, 100, kernel_size=(1, 1), stride=(1, 1)) <br />
    (2): ReLU(inplace=True) <br />
    (3): AdaptiveAvgPool2d(output_size=(1, 1)) <br />
  ) <br />
) <br />
