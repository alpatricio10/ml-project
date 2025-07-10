# Final Report
The final report can be found under report.pdf. For reference, the project guidelines were added as project_guidelines.pdf.

# Replication Steps
For GCN:
We assume that Python has already been installed. If needed, install GCN dependencies by doing the following:
1) Set up a python virtual environment called ml-project with: python3 -m venv ml-project
2) In Linux, activate the venv with: source ml-project/bin/activate
3) Install the required dependencies with: pip install -r requirements.txt
4) Register the environment as a Jupyter kernel with: python -m ipykernel install --user --name=ml-project
5) Select the new kernel before running.

To run the model:
1) Use the src/for-submission/graph_neural_network.ipynb notebook to run the final model. 
2) If not available, create a new directory called data in the same location as the models. Update the train and test dataset filename in the notebook to the appropriate location. By default, the files are named train.csv and test.csv which are located in a directory called data.
3) Run the notebook and it should output the validation scores and produce a submission file.
4) The hyperparameter tuning process is done under src/for-submission/graph_neural_network_with_optuna.ipynb instead. Follow the same steps above to run.

--------------------------------------

For other models:
Please follow the steps in src/for-submission/project.ipynb file.
1) To train the final models, you can follow the steps under "Final train and testing" section. Alternatively, already trained models can be loaded from pickle files using commands import joblib and joblib.load('../src/ens_final.pkl')
2) The hyperparameter tuning process is described under "Hyperparameter Tuning" section
3) Validation and feature importances can also be run under "Validation" and "Extract feature importances" section
