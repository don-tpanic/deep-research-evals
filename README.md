## Overview

Starting point for building deep research agent evals. The current approach proposes to build and define eval dimensions and rubrics for research agent quality based on their prompts and optionally their actual outputs.

## Implementation

### Repo structure 
* `trajectories/`: Text files store agent outputs for each research step.
* `rubrics/`: Automaically defined by calling a LLM via `python build_evals.py`
* `build_evals.py`: Given agent prompts, automatically extracts eval dimensions and define rubrics. (currently POC placeholder)
* `run_evals.py`: Given agent output, run LLM-as-a-judge against corresponding rubrics defined before.
* `inter_rater_consistency.py`: Example script of a number of inter-rater consistency methods such as weighted cohen's kappa and inter-class correlation.

### Future work
* Implement LLM interface and schema forcing for eval outputs.
* Complete the pipeline from trajectory extraction, eval building and eval execution.
* Add optional consistency checking given multiple rater outputs.

