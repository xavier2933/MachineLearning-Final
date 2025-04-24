# Transformer instructions

Code contained in Transformers.py. Model weigths stored in poorly named directories. In order to run the model trained on the transcripts, run 
```
python Transformers.py --mode interact --model_dir your_model_dir
```

In order to train a new model, run 
```
python Transformers.py --mode train --input_file ..\RawProvidedData\MITInterview\transcripts.csv --csv --model_dir transcript_model
```

You may need to install the minGPT repo, stored in a submodule in the Libraries folder. Instructions for install are in there.

Ask Xavier if you need help
