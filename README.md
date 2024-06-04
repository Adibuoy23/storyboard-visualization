## Visualizing storyboards and event boundaries
This repository contains information about using the event cognition dashboard. This installation requries Conda. Download it from [here](https://docs.anaconda.com/free/miniconda/) if not already installed on your machine. Follow these steps to set it up on your local machine. Tested with MacOs Sonoma 14.4 (M2 pro), Linux.

1. Clone this repository into your local machine. Change the destination path as needed. It should currently download this repo into your home directory.
   ```
   git clone https://github.com/Adibuoy23/storyboard-visualization.git
   ```
2. Download the data necessary for visualization. Replace the <PATH_TO_REPO> with the path location.
   ```
   cd <PATH_TO_REPO>/storyboard-visualization/data/ && curl -L https://wustl.box.com/shared/static/fm92booj4a0mcaghfap2n2oij4nfdycn.zip --output storyboard_event_boundary_visualization.zip
   ```
3. Unpack the zip file, and extract its contents in place. You should find a 'storyboard_event_boundary_visualization.csv' file in this directory after extraction.

4. Create the conda environment and install the specified packages in the requirements.txt file
   ```
   conda create -n events_dashboard python==3.9.16
   ```
5. Activate the environment
   ```
   conda activate events_dashboard
   ```
6. Install the packages in the requirements.txt using pip. Replace the <PATH_TO_REPO> with the path location.
   ```
   pip3 install -r <PATH_TO_REPO>/storyboard-visualization/requirements.txt
   ```

7. From the terminal Run the app.py file from the src directory.
   ```
   python3 <PATH_TO_REPO>/storyboard-visualization/src/app.py --path <PATH_TO_REPO>/storyboard-visualization/data/storyboard_event_boundary_visualization.csv
   ```

8. then visit http://127.0.0.1:8050 with your browser
