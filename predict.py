import utils
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

valPath = "/home/chen-ubuntu/Desktop/trainset/detectron/medline/smallTrainset/"
valjsonPath = "/home/chen-ubuntu/Desktop/train.json"

with open(valjsonPath, 'r') as f:
    print('loading json......')
    imgs_anns = json.load(f)
    print('json loader finished')

images = {}
for image in imgs_anns['images']:
    images[image['id']] = {'file_name': image['file_name'], 'annotations': []}
for ann in imgs_anns['annotations']:
    images[ann['image_id']]['annotations'].append(ann)

categories = []
for img in imgs_anns['categories']:
    categories.append(img['name'])
print('categories: ', categories)

DatasetCatalog.register("valSet", lambda I = images, P = valPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("valSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("valSet")
print('textImg_metadata: ', textImg_metadata)

cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TEST = ("valSet", )
predictor = DefaultPredictor(cfg)
output = predictor(img)

utils.draw_predImg_dicts(utils.get_textImg_dicts(images, valPath), 10, textImg_metadata, predictor)
