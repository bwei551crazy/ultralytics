from ultralytics import YOLO
#import os
#os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir = /home/yanjiaqi/miniconda3/envs/Ultralytics/lib/python3.11/site-packages/triton/backends/nvidia'

def main():

    # model = YOLO("/home/yanjiaqi/own_ultralytics/ultralytics/my_training_runs/yolo11s_bdd100k_150_orig_lbl_two datasets/weights/best.onnx")          #Change this to your custom trained model path

    # #Evaluating model inference performance    
    # results = model.predict(
    #     source = "/home/yanjiaqi/own_ultralytics/ultralytics/data/images/Tuen Mun Highway East Bound屯門公路東行 .mp4",                                     #Chuck whatever file you want for model inference here. Can be images and video
    #     conf = 0.75,                                                                    #confidence cutoff point
    #     save = True,
    #     project = "my_test_runs/yolo11s_onnx_infer",                                         #Main folder name path
    #     name = "yolo11s_bdd100k_150_orig_lbl_two_datasets",                                               #Sub folder within main folder
    #     show = True,                                                                   #Display vid during inference. Default is False
    #     verbose = False,                                                                #show progress. Default is True
    #     exist_ok = True                                                                 #Overrides if the same inferred file already exists in directory
    # ) 
    
    #Evaluating validation metrics 
    # metrics = model.val()
    # metrics.box.map                                   # map50-95
    # metrics.box.map50                                 # map50
    # metrics.box.map75                                 # map75
    # metrics.box.maps                                  # a list contains map50-95 of each category

    # results[0].show()                                 #uses matplotlib
    #"/home/yanjiaqi/own_ultralytics/ultralytics/my_training_runs/yolo11m_bdd100k_150_orig_lbl2_finetune2"
    model = YOLO("/home/yanjiaqi/own_ultralytics/ultralytics/my_training_runs/yolo11s_bdd100k_150_orig_lbl_two datasets/weights/best.pt")

    # model.export(
    #     format = "onnx",
    #     nms = True,
    #     data = "custom.yaml"
    # )

    model.export(
        format = "tflite",
        nms = True,
        int8 = True,
        fraction = 0.1,
        data = "custom.yaml",
        device = 0,
        imgsz = 640
    )

    # model = YOLO("yolo11n.pt")

    # model.export(
    #     format = "onnx",
    #     nms = True,
    #     data = "coco.yaml"
    # )

    # model.export(
    #     format = "tflite",
    #     nms = True,
    #     data = "coco.yaml"
    # )



if __name__ == "__main__":
    main()