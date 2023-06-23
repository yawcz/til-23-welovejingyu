from abstract_ai_services import AbstractDigitDetectionService, AbstractSpeakerIDService, AbstractObjectReIDService
from tilsdk.cv.types import *
from typing import Tuple, List

import numpy as np
import scipy
import cv2
import os
import librosa
import json

import torchvision.transforms as tt
from torchvision.models import resnet152, ResNet152_Weights
from torch import cat
import torch.nn as nn
import torch

import nemo.collections.asr as nemo_asr

from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.features import WaveformFeaturizer

class MockDigitDetectionService(AbstractDigitDetectionService):
    '''
    Interface for Digit Detection.
    
    This interface should be inherited from, and the following methods should be implemented.
    '''
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''

        self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_dir)
        pass

    def transcribe_audio_to_digits(self, audio_waveform: np.array) -> Tuple[int]:
        scipy.io.wavfile.write('testaudio.wav', 16000, audio_waveform.astype(np.float32))
        transcription = self.model.transcribe(paths2audio_files=['testaudio.wav'])[0][0].upper()
        digits = []
        print("TRANSCRIPTION", transcription)
        ref = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
        for i in transcription.split():
            if i in ref and ref.index(i) not in digits:
                digits.append(ref.index(i))

        return tuple(digits)
        '''
        
        Transcribe audio waveform to a tuple of ints.


        Parameters
        ----------
        audio_waveform : numpy.array
        Numpy 1d array of floats that represent the audio file. 
        It is assumed that the sampling rate of the audio is 16K.
        Returns
        -------
        results  :
        The ordered tuple of digits found in the input audio file.
        '''


class MockSpeakerIDService(AbstractSpeakerIDService):
    '''Implementation of the Speaker ID service.
    '''
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.model_dir = model_dir
        pass
    
    def identify_speaker(self, audio_waveform: np.array, sampling_rate: int) -> str:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #audio_waveform, sampling_rate=librosa.load('/content/val16k/GPTEAM_memberA_ev16k.wav')
        y_16k = librosa.resample(audio_waveform, orig_sr=sampling_rate, target_sr=16000)
        scipy.io.wavfile.write('test.wav', 16000, y_16k.astype(np.float32))
        with open('test.json', 'w') as fout:
            path = 'test.wav'
            duration = librosa.core.get_duration(filename=path)
            label = 'infer'
            metadata = {
                "audio_filepath": path,
                "duration": duration,
                "label": label
                }
            json.dump(metadata, fout)
        enrollment_manifest = 'train.json'
        test_manifest = 'test.json'
        sample_rate = 16000
        featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        dataset = AudioToSpeechLabelDataset(manifest_filepath=enrollment_manifest, labels=None, featurizer=featurizer)
        enroll_id2label = dataset.id2label
        speaker_model = EncDecSpeakerLabelModel.restore_from(restore_path=self.model_dir)
        batch_size=1
        enroll_embs, _, enroll_truelabels, _ = speaker_model.batch_inference(enrollment_manifest, batch_size, sample_rate, device=device,)
        test_embs, _, _, _ = speaker_model.batch_inference(test_manifest, batch_size, sample_rate, device=device,)
        enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
        test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))
        reference_embs = []
        keyslist = list(enroll_id2label.values())
        for label_id in keyslist:
            indices = np.where(enroll_truelabels == label_id)
            embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices)
            reference_embs.append(embedding)

        reference_embs = np.asarray(reference_embs)

        scores = np.matmul(test_embs, reference_embs.T)
        matched_labels = scores.argmax(axis=-1)
        return enroll_id2label[matched_labels[0]]

import torchvision.transforms as tt
import cv2

class BGR2RGB:
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.shape[:2])
        p_left, p_top = [(max_wh - s) // 2 for s in image.shape[:2]]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.shape[:2], [p_left, p_top])]
        return cv2.copyMakeBorder(image, p_top, p_bottom, p_left, p_right, cv2.BORDER_CONSTANT, None, value = 0)


class Resize():
    def __init__(self, output_size=(128, 128)):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        return cv2.resize(image, self.output_size, interpolation = cv2.INTER_LINEAR)

class Transforms:
    def __init__(self):
        self.transform = tt.Compose([BGR2RGB(),
                        SquarePad(),
                        Resize((128, 128)),
                        tt.ToTensor(),
                        tt.Normalize(0, 0.5)])

    def __call__(self, image):
        return self.transform(image)

from torchvision.models import resnet152, ResNet152_Weights
from torch import cat
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
        https://github.com/pytorch/examples/tree/main/siamese_network

        BCE Loss
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = resnet152(ResNet152_Weights.DEFAULT)

        for ct, child in enumerate(self.resnet.children()):
            if ct < 6:
                for param in child.parameters():
                    param.requires_grad = False


        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )


    def get_embeddings(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.get_embeddings(input1)
        output2 = self.get_embeddings(input2)
        output = cat((output1, output2), 1)
        output = self.fc(output)

        return output

class MockObjectReIDService(AbstractObjectReIDService):
    '''
    Implementation of the Object Re-ID service.
    '''
    
    def __init__(self, yolo_model_path:str, reid_model_path:str, device=None):
        self.yolo_model_path = yolo_model_path
    
        self.reid_model = torch.load(reid_model_path)
        # self.reid_model = SiameseNetwork()
        # self.reid_model.load_state_dict(torch.load(reid_model_path, map_location=torch.device('cpu')))

    def predict_image(self, model, target, img, device=torch.device('cuda')):
        reid_transforms = Transforms()

        print("COMPARING SIMLIARITY")

        xb, xb2 = reid_transforms(target).unsqueeze(0), reid_transforms(img).unsqueeze(0) # Convert to batch of 1
        model.eval()
        yb = model(xb.to(device), xb2.to(device))

        print("CONFIDENCE", yb)

        return yb
        
    
    def targets_from_image(self, scene_img, target_img) -> BoundingBox:
        if os.path.exists("runs/inference/exp/labels/scene_img.txt"):
            os.remove("runs/inference/exp/labels/scene_img.txt")
        else:
            print("The file does not exist")

        # write image to file for inference
        cv2.imwrite("scene_img.jpg", scene_img)
        # run inference
        os.system(f"python3 tools/infer.py --weights={self.yolo_model_path} --source=scene_img.jpg --save-txt")
        
        f = open("runs/inference/exp/labels/scene_img.txt", "r")

        for line in f.readlines():
            cls, x1, y1, x2, y2, conf = [float(x) for x in line.split()]
            
            plushie = scene_img[int(y1):int(y2), int(x1):int(x2)]

            match_confidence = float(self.predict_image(self.reid_model, target_img, plushie))
            plushie_class = 1 if match_confidence > 0 else 0
            
            if plushie_class:
                return BoundingBox((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        
        return None

        '''Process image with re-id pipeline and return the bounding box of the target_img.
        Returns None if the model doesn't believe that the target is within scene.

        Parameters
        ----------
        scene_img : ndarray
            Input image representing the scene to search through.

        target_img: ndarray
            Target image representing the object to re-identify.
        
        Returns
        -------
        results  : BoundingBox or None
            BoundingBox of target within scene.
            Assume the values are NOT normalized, i.e. the bbox values are based on the raw 
            pixel coordinates of the `scene_img`.
        '''
