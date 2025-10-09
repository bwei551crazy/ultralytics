from ultralytics import YOLO

def main():

    model = YOLO("rtdetr-l.pt") #Change this to your custom trained model path
    results = model.predict(
        source = "data/images/hk highway.mp4", 
        conf = 0.80, #add a parameter called conf to add confidence cutoffs
        save = True,
        project = "my_training_runs/rtdetrl_ua_detrac_infer",
        name = "rtdetrl_base_infer",
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