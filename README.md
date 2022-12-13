[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Pull Requests][pulls-shield]][pulls-url]
[![closed Pull Requests][closed_pulls-shield]][closed_pulls-url]


<p align="center">
  <h3 align="center">ITS 3D Camera-Only Detection</h3>
  <p align="center">
    Detection of cars in the Waymo-Open-Dataset
    <br />
    <a href="https://github.com/TristanBandat/ITS_3D_Camera_only_detection/issues">Report Bug</a>
  </p>
<!-- </p> -->



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Built With">Build With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#Important files and folders">Important Files and Folders</a>
      <ul>
        <li><a href="#Main file">Main file</a></li>
        <li><a href="#Training">Training</a></li>
        <li><a href="#Results">Results</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgements">Acknowledgements</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project was conducted as part of the course "KV Special Topics in AI - ITS" in WS22. <br>
The primary goal was to look at the Waymo dataset, more specifically the 3D camera-only detection <br>
part of the Motion dataset. After downloading and processing the dataset we tried 2 different models.<br>
We programmed a CNN ourselves as a model and then continued to work with the pre-built UNET. <br><br>

The UNET was developed by Olaf Ronneberger et al. for Bio Medical Image Segmentation. <br>
The model is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and <br>
does not contain any Dense layer because of which it can accept image of any size.


### Built With

* [PyCharm](https://www.jetbrains.com/pycharm/)
* [Jupyter](https://www.jupyter.com)
* [Github Desktop](https://desktop.github.com/)
* [Vim](https://www.vim.org/)
* [GCloud](https://cloud.google.com/storage/docs/reference/libraries)



<!-- GETTING STARTED -->
## Getting Started

For this project, only the training dataset was used for space reasons, which is after all also approx. 800GB in size.<br>
The downloaded records are then (or during) selected with the [extractor](extractor.py), compressed, processed <br>
and saved as tensors in a pickle file.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TristanBandat/ITS_3D_Camera_only_detection.git
   ```

2. Install dependencies<br>

   The fastest way to install the necassary dependencies is via conda: <br>
   ```shell
   conda install -f environment.yml
   ```

3. Download dataset from 
   [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
   .<br>
   For downloading the 1GB big tfrecords use the following command:<br>
   ```shell
   gcloud storage cp "[FILE]" "[FILE]" ...
   ```

4. Select and compress data to a pickle file using the [Extractor](extractor.py).<br>
   ```shell
   python extractor.py
   ```

5. Now one can proceed with the [notebook](ITS_3D_Camera_only_challenge.ipynb) or the [main](main.py) file.
   

<!-- FILES & FOLDERS -->
## Important files and folders

### Main file
[main.py](main.py)<br>
The entry point is the main.py file. Here one can find all the different hyperparameters and available models to train.<br>
Furthermore the path to the dataset and the final models is also chosen here.<br>
If all the necessary packages are installed one can simply run the file and the training starts.<br>

### Training
[train.py](train.py)
Here one can find the whole project structure. Further details and explanations are contained in the
[notebook](ITS_3D_Camera_only_challenge.ipynb).

### Results

In the `results/` folder the final model and the plots can be found. With the help of tensorboard one can also <br>
view additional plots like train/validation loss or the gradients.


<!-- CONTACT -->
## Contact

Tristan Bandat - tristan.bandat@gmail.com <br>
Philipp Meingaßner - meingassner.p@gmail.com <br>
Jakob Eggl  <br>
Florian Hitzler  <br>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TristanBandat/ITS_3D_Camera_only_detection.svg?style=for-the-badge
[contributors-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TristanBandat/ITS_3D_Camera_only_detection.svg?style=for-the-badge
[forks-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/network/members
[stars-shield]: https://img.shields.io/github/stars/TristanBandat/ITS_3D_Camera_only_detection.svg?style=for-the-badge
[stars-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/TristanBandat/ITS_3D_Camera_only_detection.svg?style=for-the-badge
[issues-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/issues
[pulls-shield]: https://img.shields.io/github/issues-pr/TristanBandat/ITS_3D_Camera_only_detection.svg?style=for-the-badge
[pulls-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/pulls
[closed_pulls-shield]: https://img.shields.io/github/issues-pr-closed/TristanBandat/ITS_3D_Camera_only_detection?style=for-the-badge
[closed_pulls-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/pulls?q=is%3Apr+is%3Aclosed
