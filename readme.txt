Welcome, editors and reviewers.

This is a temporary repository for review purpose, corresponding to the journal article manuscript titled "Determine PV-Battery Dynamic Firm Capacity Using Deep Reinforcement Learning".

data/ contains the data used in this study.
code/ contains the Python source code that implements the experiment.

The experiment is scripted in a Jupyter Notebook (ipynb) file seeded_experiments.ipynb in code/ directory.
A lighter weight example script example_experiment.ipynb provides a quick example of how the experiment is conduceted.

The ipynb scripts depend on the following files:
    code/definitions.py
    code/config.py
    code/helper_fns.py
    code/application.py
    code/dfc_gymnasium/*

# Run the Experiment 
The experiment can occupy more than 3 GB memory.
Change working directory to code/ before running .ipynb or .py files.
Running the seeded_experiments.ipynb script will result in the following output:
    evaluation/
    models/
    evaluation_journal*.csv
    runid_seed_log.csv
    A temporary wandb/ folder will be created by wandb, which tracks the training status.

At the end of the ipynb script, these output can be archived as a .zip file saved in ../results/ 
Optionally, the script can purge the archived content and get ready for the next experiment.

In addition to the experiment code, there are also some Python scripts in code/scripts/
The experiment does not depend on these scripts to run, but one may find them useful when building applications other than the experiment discussed in the manuscript.

