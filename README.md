# Auto Nuclei Segmentation
## Usage
This program will automatically create masks and polygons around nucleis present in nuclei images.
Support labelme polygons annotation. 
## Requirement
* Python 3.7 or higher
## Installation
```bash
git clone https://github.com/quandapro/auto_segmentation
cd auto_segmentation
python -m pip install -r requirements.txt
```
## How to use
* Image files are stored in data/images folder
* To generate masks and labelme annotation
```Bash
python predict.py
```
* Open data/images folder on labelme and edit polygons 
