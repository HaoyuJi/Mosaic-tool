# Mosaic-tool
An automatic mosaic tool

一个用于自动检测人头并打码的工具



## 介绍
- 能够批量检测一个input文件夹里的所有视频文件，并将其批量人头打马赛克并输出到output文件夹里
- headTracker.py就是经典的实现上述功能
- headTracker_cheat.py可以用一个cheat视频文件来检测人头，但是打马赛克是打到input文件夹里(比如你有一些已经跑过人体关键点检测或者面部检测的视频，由于遮挡导致面部打马赛克可能会效果不好，就可以用这个来检测原视频的人头框，并打马赛克到你处理过的视频里)

  

## 运行
测试一下： 
地址记得改成自己想放的位置

For headTracker.py, run:
```
python headTracker.py --input_path D:/assessment/original/ --output_path D:/assessment/detect/
```
For headTracker_cheat.py, run:
```
python headTracker_cheat.py --input_path D:/assessment/original/ --output_path D:/assessment/detect/
```
