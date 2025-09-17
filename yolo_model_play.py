from ultralytics import YOLO
import cv2

#will use the yolov5su.pt pretrained model rather than the original yolov5s.pt since this code uses straight from the ultralytics repo, rather than the yolov5 repo

#loading a pretrained model
model = YOLO("my_training_runs/yolov5su_1.pt3/weights/best.pt") 

#=================================================Training code==========================================================#
# # Looks like training already does validation for you as well. 
# # train_results only contains training results/metadata, not the entire model itself. By doing model.train, I'm already updating the preloaded model with its pretrained weights. 
train_results = model.train(
    data="coco128.yaml",
    epochs = 300,
    imgsz = 640,
    batch = 24,
    lr0 = 0.01,
    project = "my_training_runs",
    name = "yolov5su_further_300",
    save = True,
    exist_ok = False
)

#========================================Purely validation code====================================================#
#Only used if you just want to do purely validation. Otherwise, use train()
#metrics = model.val()


#-------------------------At this point, model is already updated with the new weights from the training above (if model.train() was runned)-----------------------------#

#=====================================================Format on doing vid inference================================================#
# model.predict(
#     source = input_vid,
#     conf = 0.5,
#     save = True,                                #save vid to specified folder
#     project = save_folder,
#     name = name_of_inferred_vid,
#     show = True,                                #display vid during processing
#     verbose = True                              #show progress
# )

#===========================================Format for doing image inference=============================================================#

#Doing model("path/to/image", conf = 0.7) and model.predict(source = "path/to/image", conf = 0.7) produce same result

results = model.predict(source = "data/images/queenstreetnz.jpeg", conf = 0.7) #add a parameter called conf to add confidence cutoffs
results[0].show() #uses matplotlib
annotated_frame = results[0].plot()

cv2.imshow("Yolov5u detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("my_training_runs/yolov5su_further_300/queenstreetnz_0.7.jpg", annotated_frame)
print("Saved")

#path = model.export(format="onnx")