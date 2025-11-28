# Lab1实验报告

> 黄嘉诚 22302010051

## 设计思路

恒温器代表两个state，cooling与heating，故而只需要设计一个有此二状态的FSM即可，初始状态参照实验资料书设定为heating：

![image-20250928051023080](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928051023080.png)

## 实现过程

使用Ptolemy实现该FSM，需要使用到一个CompositeActor与一个ModalModel，为了仿真与观察数据，再加上两个SequencePlotter，综上，在SDF Director下运行，即：

![image-20250928051225027](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928051225027.png)

其中，FSMActor展开为：
![image-20250928051309073](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928051309073.png)

CompositeActor展开为：

![image-20250928051336137](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928051336137.png)

## 运行结果

在上图所示的参数设置下，设定仿真时间单位为200，可以得到这样的数据：

![image-20250928051517953](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928051517953.png)

左图为加热率随时间变化折线图，右图为温度随时间变化散点图，可以见到，在0.8与1.4这两个加热率变化的节点，经过一段时间后温度才会明显变化。

## 参数分析

- 降低`heatOnThreshold`

![image-20250928051850920](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928051850920.png)

可见降低打开暖气的阈值会让温度低的时间更长，很符合直觉

- 提高`heatOffThreshold`

![image-20250928052051832](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928052051832.png)

同样，提高关闭暖气的阈值，温度高的时间也变长了

- 提高`heatingRate`

![image-20250928052502512](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928052502512.png)

提高加热率就会让温度达到暖气关闭阈值的时间变短，图中也有体现

- 提高`coolingRate`的绝对值

![image-20250928052859908](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20250928052859908.png)

同理，达到暖气打开阈值的时间也会提前