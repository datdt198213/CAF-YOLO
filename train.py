# from ultralytics import YOLO
# import multiprocessing

# yaml会自动下载
# def main():
#     model = YOLO("E:/ultralytics-main/ultralytics/cfg/models/v8/yolov8-CAFMAttention.yaml")  # build a new model from scratch
# # model = YOLO("d:/Data/yolov8s.pt")  # load a pretrained model (recommended for training)

#         # Train the model
#     results = model.train(data="dataset/blood.yaml", batch=8,epochs=300, imgsz=640)

# if __name__=='__main__':
#     multiprocessing.freeze_support()
#     main()

from ultralytics import YOLO
import multiprocessing

def main():
    # model = YOLO("/content/CAF-YOLO/ultralytics/cfg/models/v8/yolov8-CAFMAttention.yaml")  # build a new model from scratch
    # model = YOLO("/content/CAF-YOLO/runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)
    model = YOLO("/content/best_game.pt")  # load a pretrained model (recommended for training)
        # Train the model
    results = model.train(data="/content/datasets/game_class.yaml", epochs=30, imgsz=640)

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
