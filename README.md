# biomed_translation
This github linked to our work evlauating the impact of English translation on biomedical information extraction from French medical notes.

The objective of our experiments was to evaluate the impact of English translation on biomedical named-entity recognition and normalization. The work is divided in three axes : 
  - a native French axis : developped in Axes/axis1. With first a NER step performed by the model : https://github.com/percevalw/nlstruct with a Fasttext algorithm trained from scratch on french biomedical notes and a CamemBERT-large _fine-tuned_
https://github.com/percevalw/nlstruct
