# A* line segmenter
## prepare Environment
You need to have the anaconda environment manager installed on your computer. 
If so, run the command 
```code
conda env create -f environment.yml
```

and acrivate the environment: 
```code
conda activate AstarSegm
```
# Run the example

the file ```line_segmenter.py``` allows to perform the line segmentation method. Run it with the command:
```code
python line_segmenter.py
```


runining the script, the images stored in the folder "input_data\examples" are segemnted.
The output of segmentation is saved in a new folder "lines"