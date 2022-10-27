# TART Imaging Pipeline (TIP)
An imaging pipeline for the TART interferometer. The pipeline allows for testing via simulated sky models and interferometer positioning models.

The tool was developed as a guide for how to create your own imaging pipeline. It should be utilised by students in order to aid in the teaching of radio interferometry.

The pipeline also allows for the imaging process of the TART telescope <https://tart.elec.ac.nz/signal/home> in New Zealand and in South Africa, however the South African interferometer requires a VPN to access. The pipeline receives visibilities from the interferometer and applies the steps of radio interferometry to create visible images of the sky above them at that point. This pipeline
does not use calibration or deconvolution in the process of imaging.

## Running
Python 3 must be installed as well as the requirements listed below.

[comment]: <> (Latex must also be installed for the image axes. The following are recommended)

[comment]: <> (* texlive-latex-extra)

[comment]: <> (* texlive-fonts-recommended)

[comment]: <> (* dvipng)

[comment]: <> (* cm-super)


Simply run the run.bat or run.sh depending on your operating system.

### Requirements
* numpy
* scipy
* matplotlib
* cherrypy
* jinja2
* pillow
