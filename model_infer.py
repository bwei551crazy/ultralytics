from ultralytics import YOLO

def main():

    model = YOLO("my_training_runs/rtdetr_ua_detrac_10k_100/weights/best.pt") #Change this to your custom trained model path
    results = model.predict(
        source = "data/images/20250916_190547.mp4", 
        conf = 0.20, #add a parameter called conf to add confidence cutoffs
        save = True,
        project = "my_test_runs/rtdetrl_infer",
        name = "rtdetrl_ua-detrac_infer",
        #show = True,                                #display vid during processing
        #verbose = True,                              #show progress
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