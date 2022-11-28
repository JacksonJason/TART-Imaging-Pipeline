# TART Imaging Pipeline (TIP)
An imaging pipeline for the Transient Array Radio Telescope (TART). The pipeline allows for testing via simulated sky models and interferometer positioning models.

The tool was developed as a guide for how to create your own imaging pipeline. It should be utilised by students in order to aid in the teaching of radio interferometry.

The pipeline also allows for the imaging process of the TART telescope <https://tart.elec.ac.nz/signal/home> in New Zealand and in South Africa, however the South African interferometer requires a VPN to access. The pipeline receives visibilities from the interferometer and applies the steps of radio interferometry to create visible images of the sky above them at that point. This pipeline does not use deconvolution in the process of imaging.

**Note**: As of 2022, the SA Stellenbosch TART appears to be offline and we have been unable to reach the people responsible for maintaining it. The NZ TART still appears to work however.

## Running
Python 3 must be installed as well as the requirements listed below.

Simply run the run.bat or run.sh depending on your operating system.

### Requirements
* numpy
* scipy
* matplotlib
* cherrypy
* jinja2
* pillow

* Latex must also be installed for the image axes. The following are recommended:
    * texlive-latex-extra
    * texlive-fonts-recommended
    * dvipng
    * cm-super

 For any questions or comments please direct them to [Jason Jackson](mailto:ajsnpjackson@gmail.com).
