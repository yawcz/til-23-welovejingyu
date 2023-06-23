from dataclasses import dataclass, field
from tilsdk.localization.types import *
from tilsdk.cv.types import *
from typing import Any, List, Union
from datetime import datetime
import cv2
import json
import base64

# @dataclass
class Report: 
    id : str
    timestamp : datetime
    image : Any #cv2 image 
    pose : RealPose
    checkpoint_id : int
    
    time_valid: bool = False
    situation : str = None
    audio : str 
    digits : int
    score : int


    # @staticmethod
    # def from_json(s:Union[str,bytes], id:str, timestamp:datetime=datetime.now()):
    #     data = json.loads(s)

    #     img_data = base64.b64decode(data['image'])
    #     np_img = np.asarray(bytearray(img_data), dtype=np.uint8)
    #     image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    #     report = Report(
    #         id = id,
    #         timestamp=timestamp,
    #         pose = RealPose(**data['pose']),
    #         image = image,
    #         targets = [DetectedObject(
    #             id = t['id'],
    #             cls = t['cls'],
    #             bbox = BoundingBox(**t['bbox'])
    #         ) for t in data['targets']]
    #     )
    #     return report


    # def get_annotated(self, thickness:int=1):
    #     '''Get image with target bboxes drawn.'''
    #     img = self.image.copy()
    #     for target in self.targets:
    #         if target.cls == 0:
    #             color=(0,0,255) # red
    #         elif target.cls == 1:
    #             color=(0,255,0) # green
    #         else:
    #             color=(255,0,0) # blue, unknown
    #         bbox = target.bbox
    #         pt1 = (round(bbox.x-bbox.w/2), round(bbox.y-bbox.h/2))
    #         pt2 = (round(bbox.x+bbox.w/2), round(bbox.y+bbox.h/2))
    #         img = cv2.rectangle(self.image, pt1, pt2, color=color, thickness=thickness)
    #         img = cv2.putText(img, '{}'.format(target.cls), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)

    #     return img


if __name__ == '__main__':
    report = Report()