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
    <li><a href="#Important files and folders">Important Files and Folders</a></li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgements">Acknowledgements</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore 
magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing 
elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos 
et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem 
ipsum dolor sit amet.


### Built With

* [PyCharm](https://www.jetbrains.com/pycharm/)
* [Github Desktop](https://desktop.github.com/)
* [Vim](https://www.vim.org/)



<!-- GETTING STARTED -->
## Getting Started

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore 
magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TristanBandat/ITS_3D_Camera_only_detection.git
   ```

2. Install dependencies<br>

   The fastest way to install the necassary dependencies is via conda: <br>
   `conda install -f environment.yml`<br><br>

3. Download dataset from ...<br><br>
   

<!-- FILES & FOLDERS -->
## Important files and folders

### main.py
The entry point is the main.py file. Here one can find all the different hyperparameters and available models to train.
Furthermore the path to the dataset and the final models is also chosen here.
If all the necessary packages are installed one can simply run the file and the training starts.

### train.py

Here one can find the whole project structure. Further details and explanations are contained in the notebook [ITS_3D_Camera_only_challenge](ITS_3D_Camera_only_challenge.ipynb).

### results

In the results folder the final model and the plots can be found. With the help of tensorboard one can also view additional plots like train/validation loss or the gradients.


<!-- CONTACT -->
## Contact

Tristan Bandat - tristan.bandat@gmail.com <br>
Philipp Meingaßner - meingassner.p@gmail.com <br>
Jakob Eggl -  <br>
Florian Hitzler -  <br>



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
[license-shield]: https://img.shields.io/github/license/TristanBandat/ITS_3D_Camera_only_detection.svg?style=for-the-badge
[license-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/blob/master/LICENSE.txt
[closed_pulls-shield]: https://img.shields.io/github/issues-pr-closed/TristanBandat/ITS_3D_Camera_only_detection?style=for-the-badge
[closed_pulls-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/pulls?q=is%3Apr+is%3Aclosed
[closed_issues-shield]: https://img.shields.io/github/issues-closed/TristanBandat/ITS_3D_Camera_only_detection?style=for-the-badge
[closed_issues-url]: https://github.com/TristanBandat/ITS_3D_Camera_only_detection/issues?q=is%3Aissue+is%3Aclosed
