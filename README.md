# biomed_translation
This github linked to our work evlauating the impact of English translation on biomedical information extraction from French medical notes.

The objective of our experiments was to evaluate the impact of English translation on biomedical named-entity recognition and normalization. The work is divided in three axes : 
  - a native French axis (axis1 in Figure below)  : developped in Axes/axis1. 
    1. First a NER step is performed by the model : https://github.com/percevalw/nlstruct with a Fasttext algorithm trained from scratch (https://fasttext.cc/) on french biomedical notes and a CamemBERT-large (https://camembert-model.fr/) _fine-tuned_. Then, we performed a normalization step with two different algorithms: https://github.com/percevalw/deep_multilingual_normalization and 

![Overall_diagram(1)](https://user-images.githubusercontent.com/81175825/174573508-fded2955-5282-42dc-83b0-3d852f240085.png)
