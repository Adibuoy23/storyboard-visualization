## Visualizing storyboards and event boundaries
This repository contains information about using event cognition dashboard. This installation requries Conda. Download it from [here](https://docs.anaconda.com/free/miniconda/) if not already installed on your machine. Follow these steps to set it up on your local machine. Tested with MacOs, Linux.

1. Clone this repository into your local machine. Change the destination path '/storyboard-visualization/' as needed. It should currently download this repo into your home directory.
   ```
   git clone https://github.com/Adibuoy23/storyboard-visualization.git
   ```
2. Download the data necessary for visualization
   ```
   cd <PATH TO THE REPO>/storyboard-visualiation/data/ && curl -L https://wustl.box.com/shared/static/fm92booj4a0mcaghfap2n2oij4nfdycn.zip --output storyboard_event_boundary_visualization.zip
   ```
3. Unpack the zip file, and extract its contents in place. You should find a 'storyboard_event_boundary_visualization.csv' file in this directory after extraction.

4. Create the conda environment and install the specified packages in the requirements.txt file
   ```
   conda create --name events_dashboard python==3.9 pandas numpy plotly dash-player
   ```
5. Activate the environment
   ```
   conda activate events_dashboard
   ```   

7. From the terminal Run the app.py file from the src directory.
   ```
   python3 /storyboard-visualzation/src/app.py
   ```

8. then visit http://127.0.0.1:8050 with your browser
