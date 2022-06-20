# biomed_translation
This github is linked to our work evlauating the impact of English translation on biomedical information extraction from French medical notes.

The objective of our experiments was to evaluate the impact of English translation on biomedical named-entity recognition and normalization. The work is divided in three axes : 
  - a native French axis (axis1 in Figure below)  : developped in Axes/axis1. 
    1. First a NER step is performed by the model : https://github.com/percevalw/nlstruct with a Fasttext algorithm trained from scratch (https://fasttext.cc/) on french biomedical notes and a CamemBERT-large (https://camembert-model.fr/) _fine-tuned_. 
    2. Then, we performed a normalization step with two different algorithms: the deep multilingual normalization (https://github.com/percevalw/deep_multilingual_normalization) and the CODER algorithm (https://github.com/GanjinZero/CODER). 
  
  - two translated English axes developped in Axes/axis2.1 and Axes/axis2.2 directories:both first with a translation step performed by the opus-mt-fr-en algorithm (https://huggingface.co/Helsinki-NLP/opus-mt-fr-en) fine-tuned on a biomedical dataset (https://paperswithcode.com/dataset/wmt-2016-biomedical). 
Then, axis2.1 is done with a NER step and a normalization step with the same algorithms as before. 
axis2.2 uses the MedCAT algorithm which performs both NER and normalization at the same time. 

### Axes/axis1 to axis2.2 include overall branch, whereas each folder : NER, normalization, translation contains the intermediate steps 
    

![Overall_diagram(1)](https://user-images.githubusercontent.com/81175825/174573508-fded2955-5282-42dc-83b0-3d852f240085.png)
