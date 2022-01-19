import torch
import pyredner
import sys
import redner
import os
import random
import gc
from torchvision.utils import save_image
gc.collect()

torch.cuda.empty_cache()
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"
#print(device)

#pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_device(torch.device('cuda:2'))
class ShapenetRender():

    def __init__(self,b1,b2,b3):
        # self.inp_dir_path = inp_dir_path
        # self.out_path = out_path
        # self.path = os.path.join(os.getcwd(),self.inp_dir_path)
        # self.model_path_list = [self.path]
        # #print(self.model_path_list[0])
        self.b1,self.b2,self.b3 = int(b1),float(b2),int(b3)
        self.cam_pos_dict = {0 : torch.tensor([0.0,0.0,-1.0 * self.b2]),
                             1 : torch.tensor([0.0,1.0*self.b2,0.0]),
                             2 : torch.tensor([1.0*self.b2,0.0,0.0]),
                             3 : torch.tensor([0.0,0.5*self.b2,-1.0 * self.b2]),
                             4 : torch.tensor([1.0*self.b2,0.5*self.b2,0.0]),
                             5 : torch.tensor([0.5*self.b2,0.0,-1.0 * self.b2]),
                            }
        self.cam_up_dict = {0 : torch.tensor([0.0,1.0,0.0]),
                             1 : torch.tensor([0.0,0.0,-1.0]),
                             2 : torch.tensor([0.0,1.0,0.0]),
                             3 : torch.tensor([0.0,1.0,0.0]),
                             4 : torch.tensor([0.0,1.0,0.0]),
                             5 : torch.tensor([0.0,1.0,0.0]),
                            }
        self.light2_up_dict = {0 : torch.tensor([0.0,0.0,-5.0]),
                             1 : torch.tensor([0.0,0.0,-5.0]),
                             2 : torch.tensor([5.0,0.0,0.0]),
                             3 : torch.tensor([0.0,0.0,-5.0]),
                             4 : torch.tensor([5.0,0.0,0.0]),
                             5 : torch.tensor([5.0,0.0,0.0]),
                            }
        self.light1_up_dict = {0 : torch.tensor([0.0, 5.0, 0.0]),
                             1 : torch.tensor([0.0, 5.0, 0.0]),
                             2 : torch.tensor([0.0, 5.0, 0.0]),
                             3 : torch.tensor([0.0, 5.0, 0.0]),
                             4 : torch.tensor([0.0, 5.0, 0.0]),
                             5 : torch.tensor([0.0, 0.0, -5.0]),
                            }
        self.intensity_dict = {0 : torch.tensor([10000.0, 10000.0 , 10000.0]),
                               1 : torch.tensor([0.0, 10000.0 , 10000.0]),
                               2 : torch.tensor([10000.0, 0.0 , 10000.0]),
                               3 : torch.tensor([10000.0, 10000.0 , 0.0]),
                            }





    def render_acp(self,inp_path,out_path):
        objects,camera,light1,light2,scene,img = None,None,None,None,None,None
        try:
            objects = pyredner.load_obj(inp_path, return_objects=True)[:-1]
        except:
            print("Error in load object {}".format(inp_path))
            return
        try:
            camera = pyredner.Camera(position = self.cam_pos_dict[self.b1],
                                look_at = torch.tensor([0.0, 0.0, 0.0]),
                                up = self.cam_up_dict[self.b1],
                                fov = torch.tensor([60.0]), # in degree
                                clip_near = 1e-2, # needs to > 0
                                resolution = (224, 224),
                                )
            #camera = pyredner.automatic_camera_placement(objects, resolution=(480, 640))

            light1 = pyredner.generate_quad_light(position = self.light1_up_dict[self.b1],
                                                look_at = torch.zeros(3),
                                                size = torch.tensor([0.1, 0.1]),
                                                intensity = self.intensity_dict[self.b3])

            light2 = pyredner.generate_quad_light(position = self.light2_up_dict[self.b1],
                                                look_at = torch.zeros(3),
                                                size = torch.tensor([0.1, 0.1]),
                                                intensity = self.intensity_dict[self.b3])

            scene = pyredner.Scene(camera = camera, objects = objects+[light1,light2])
        except:
            print("Error in variable setting")
            return
        try:
            img = pyredner.render_pathtracing(scene,num_samples = (128,4))
        except:
            print("Error in rendering the image")
            return
        try:
            pyredner.imwrite(img.cpu(),out_path+".png")
            #print(out_path)
        except:
            print("Error in writing the image ",inp_path)
            return

    # def render_random(self,num_samples):
    #     for model_path in self.model_path_list:
    #         #print(model_path)
    #         model_list = [os.path.join(model_path,f) for f in os.listdir(model_path) 
    #                         if not os.path.isfile(os.path.join(model_path,f))]
    #         model_list_random = random.sample(model_list,min(num_samples,len(model_list)))
    #         for model in model_list_random:
    #             #print(model_path.split("/")[-1])
    #             #print(model.split("/")[-1])
    #             self.render_acp(model+"/models/model_normalized.obj",
    #                             self.out_path+"/"+model.split("/")[-1])
    




# if __name__ == "__main__":
#     shapenetrender = ShapenetRender(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
#     shapenetrender.render_random(int(sys.argv[6]))