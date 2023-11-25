from steps.config import CFG

class Model:
    def __init__(self,name: str):
        self.Model = None
        if name == "VGG16":
            from steps.Models.VGG16 import VGG16
            self.Model = VGG16(CFG.num_classes)
        if name == "ResNet34":
            from steps.Models.ResNet import BasicBlock
            from steps.Models.ResNet import ResNet
            layers=[3, 4, 6, 3]
            self.Model = ResNet(BasicBlock, layers,CFG.num_classes)
            print(self.Model)
        if name == "InceptionV3":
            from steps.Models.Inceptionv3 import InceptionV3
            self.Model = InceptionV3()
        if name == 'googlenet':
            from steps.Models.googlenet import GoogleNet
            self.Model = GoogleNet(CFG.num_classes)
            
    def get_model(self):
        return self.Model
