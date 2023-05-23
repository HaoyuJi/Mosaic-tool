import cv2
import argparse
from headDetect.models.experimental import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Video file for evaluation.')
    parser.add_argument('--output_path', type=str, help='Video file for evaluation.')

    return parser.parse_args()

class headDetection():
    def __init__(self, weights = './headDetect/weights/helmet_head_person_s.pt', imgsz=640, device='0', conf_thres=0.4, iou_thres=0.5, classes=None, agnostic_nms=False):

        # Initialize
        self.device = torch_utils.select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    def inference(self, frame):
        img0 = frame

        output_det = []
        with torch.no_grad():
            img = self.letterbox(img0, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # output_det.append([])
                    for d in det:
                        if d[-1] == 1 or d[-1] == 2:
                            # output_det[-1].append({'frame': iter, 'bbox': (d[:-2]).tolist(), 'conf': d[-2]})
                            output_det.append((d[:-2]).tolist())

        return output_det

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

# 自定义一个函数来获取文件夹中所有视频文件的路径
def get_video_paths(folder_path):
    # 定义我们要查找的视频文件扩展名列表
    video_extensions = ['.mp4', '.avi', '.mkv']
    # 创建一个空列表，用于存储找到的视频文件路径
    video_paths = []

    # 使用os.walk遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历每个文件
        for file in files:
            # 检查文件是否以我们定义的任何视频扩展名结尾
            if any(file.endswith(ext) for ext in video_extensions):
                # 如果是视频文件，将其路径添加到video_paths列表中
                video_paths.append(os.path.join(root, file))

    # 返回找到的所有视频文件路径
    return video_paths


if __name__ == '__main__':

    args = parse_args()
    # 指定要搜索的文件夹路径
    folder_path = args.input_path
    # 调用get_video_paths函数并传入文件夹路径，获取所有视频文件路径
    video_paths = get_video_paths(folder_path)

    for video in video_paths:

        videoname = video.split("/")[3][0:-4]  # 给当前读到的video起个名

        cap = cv2.VideoCapture(video)
        headModel = headDetection()

        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 创建一个 VideoWriter 对象，用于将处理后的帧写入新的视频文件
        videoname_avi = videoname + ".avi"
        outvideo = cv2.VideoWriter(args.output_path + videoname_avi, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (frame_width, frame_height))

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if ret:

                headBoxes = headModel.inference(frame.copy())


                if len(headBoxes):
                    #头部检测（画框）
                    for box in headBoxes:
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

                    #头部遮挡（打马赛克）
                    for box in headBoxes:
                        # 获取头部区域
                        head = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        # 下采样，降低分辨率
                        head = cv2.resize(head, (int((box[2]-box[0])/20), int((box[3]-box[1])/20)))
                        # 上采样，提升分辨率
                        head = cv2.resize(head, (int(box[2]-box[0]), int(box[3]-box[1])), interpolation=cv2.INTER_NEAREST)
                        # 应用马赛克
                        frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = head

            else:
                break

            # 将处理后的帧写入新的视频文件
            outvideo.write(frame)

            # 记录处理结束时间
            end_time = time.time()
            # 计算处理时间并打印到控制台
            processing_time = end_time - start_time
            print("Processing time for this frame: {:.4f} seconds.".format(processing_time))
            # key = cv2.waitKey(1)
            #
            # if key == 27 or key == ord('q'):
            #     break

        cap.release()