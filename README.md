
# Floor Segmentation

Deep Learning based floor segmentation model that may be used to make floor pixels transparent. A useful tool for deploying and testing new floor designs in real time.
## Demo
<table>
  <tr>
    <td>Image</td>
     <td>Mask</td>
     <td>Result</td>
  </tr>
  <tr>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/images/3.jpg" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/masks/3.png" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/results/3.png" width="200" height="150"0></td>
  </tr>
  <tr>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/images/1.jpg" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/masks/1.png" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/results/1.png" width="200" height="150"0></td>
  </tr>
  <tr>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/images/4.jpg" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/masks/4.png" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/results/4.png" width="200" height="150"0></td>
  </tr>
  <tr>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/images/7.jpg" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/masks/7.png" width="200" height="150"></td>
    <td><img src="https://github.com/uzairmughal20/Floor-Segmentation/blob/master/results/7.png" width="200" height="150"0></td>
  </tr>
 </table>
 
## Installation

```bash
    conda create --name floor_segmentation
    conda activate floor_segmentation
    pip install -r requirements.txt
```
    
## Running Tests

To run tests, run the following command

```bash
    python floor_segmentation.py "image_path" 
    example: python floor_segmentation.py ./images/1.jpg
```


## Training
## Dataset
[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
## Used By

This project is used by the following company:

- Shiftwave Technologies


## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)


## References

 - [Segmentation Models](https://github.com/qubvel/segmentation_models)
