
## **Layers currently supported:** <br/>
* Conv2d & 3d <br/>
* Linear <br/>
* BatchNorm2d & 3d <br/>
* Relu <br/>
* MaxPooling & AVGPooling 2d & 3d <br/>
* Softmax <br/>

## **Testing:** <br/>
to test it with your net, first import the profiler: "from profileCNN import NN_Profiler" <br/>
then call the profiler after specifying model and input size: "NN_Profiler(model, inp_size)"<br/>
this will return total number of operations and parameters; it can be also checked for layer wise calculations<br/> 

A test example, "test.py", is added for alexnet and 3D CNN example from https://github.com/HHTseng/video-classification

### otherwise under dev <br/>
@smoradi
