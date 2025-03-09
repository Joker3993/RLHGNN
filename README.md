# RLHGNN

## Usage
The following commands can be directly run to process the six datasets in the paper:
- 'bpi13_closed_problems'
- 'bpi13_problems'
- 'bpi13_incidents'
- 'bpi12w_complete'
- 'bpi12_all_complete'
- 'BPI2020_Prepaid'

### Preprocessing Event Logs

The preprocessing code for event logs is located in the `data.process.py` script.

### Training the Basic Performance Model for Graph Configurations

By running the `main.py` script, the operation for four basic graph configurations can be completed. The corresponding evaluation metrics can be calculated through the `metrics.py` script.

### Training the DQN Decision Model

By running the `env_train.py` script, the training of the decision model can be completed, and the graph data for the final hybrid graph configuration can be generated.


### Training the Final Next Activity Prediction Model

Run the `final_main.py` script to train the prediction model. Calculate the metrics through the `metrics_final.py` script.

## Tools
pytorch: Used for deep learning operations.
python: The programming language used for the project.
## Data
The event logs for predictive business process monitoring can be found at 4TU Research Data.