from ultralytics import YOLO

def main():

    model = YOLO("my_training_runs/yolo11m_ua_detrac_10k_502/weights/best.pt") #Change this to your custom trained model path
    results = model.predict(
        source = "datasets/UA-DETRAC-10K-SAMPLE.v1i.yolov11/test/images", 
        conf = 0.2, #add a parameter called conf to add confidence cutoffs
        save = True,
        project = "my_training_runs/yolo11m_ua_detrac_infer",
        name = "ua_detrac_10k_ifer_low_con",
        #show = True,                                #display vid during processing
        verbose = True,                              #show progress
        exist_ok = True
    ) 
    
    # metrics = model.val()
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category

    # results[0].show() #uses matplotlib

if __name__ == "__main__":
    main()