import cv2, numpy as np 
out = cv2.VideoWriter('data/videos/test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,480)) 
[out.write(np.random.randint(0,255,(480,640,3),dtype=np.uint8)) for i in range(250)] 
out.release() 
print('Test video created') 
