import os
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
import os.path

@DATASET_REGISTRY.register()
class cifar10(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """
    dataset_dir = "cifar10"
    def __init__(self, cfg):

        self.clsname = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships' ,'trucks']

        train_x, test, val = self.get_Data()
        if len(val) == 0:
            val = None
        super().__init__(train_x=train_x,  val=val, test=test)

    def get_Data(self):
        train_x= []
        test = []
        val = []
        classnamelist = self.clsname
        for i,j,k in os.walk(os.path.join(self.cfg.DATASET.ROOT,'OOD','cifar10','train')):
            for item in k:
                imgpath = os.path.join(i,item)
                label = i.split('/')[-1]
                item = Datum(impath=imgpath, label=int(label), classname=classnamelist[int(label)])
                train_x.append(item)
        for i,j,k in os.walk(os.path.join(self.cfg.DATASET.ROOT,'OOD','cifar10','train')):
            for item in k:
                imgpath = os.path.join(i,item)
                label = i.split('/')[-1]
                item = Datum(impath=imgpath, label=int(label), classname=classnamelist[int(label)])
                test.append(item)
                val.append(item)
        return train_x, test, val


@DATASET_REGISTRY.register()
class cifar100(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar100"

    def __init__(self,cfg):
        self.cfg = cfg
        self.clsname = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle','bottle', 'bowl', 'boy','bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair','chimpanzee','clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin','elephant','flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp','lawn_mower','leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain','mouse', 'mushroom','oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree','plain', 'plate','poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea','seal', 'shark','shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar','sunflower','sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train','trout', 'tulip','turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        train_x, test, val = self.get_Data()

        if len(val) == 0:
            val = None

        super().__init__(train_x=train_x,  val=val, test=test)

    def get_Data(self):
        train_x= []
        test = []
        val = []
        classnamelist = self.clsname
        for i,j,k in os.walk(os.path.join(self.cfg.DATASET.ROOT,'OOD','cifar100','train')):
            for item in k:
                imgpath = os.path.join(i,item)
                label = i.split('/')[-1]
                item = Datum(impath=imgpath, label=int(label), classname=classnamelist[int(label)])
                train_x.append(item)
        for i,j,k in os.walk(os.path.join(self.cfg.DATASET.ROOT,'OOD','cifar100','test')):
            for item in k:
                imgpath = os.path.join(i,item)
                label = i.split('/')[-1]
                item = Datum(impath=imgpath, label=int(label), classname=classnamelist[int(label)])
                test.append(item)
                val.append(item)
        return train_x, test, val
