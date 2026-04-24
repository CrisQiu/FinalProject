# {{ cookiecutter.project_name }}

**Author:** {{ cookiecutter.author_name }}  
**Contact:** {{ cookiecutter.email }}
**Organization:** {{ cookiecutter.orgname }}
**Website:** {{ cookiecutter.url }}

## Project Overview
{{ cookiecutter.description }}
Breifly describe your research for anyone who finds this repository before reading your manuscript.* problem statement: why does your research matter
* problem statement: what overarching theoretical, applied, societal, environmental problem does your research address? 
* challenge statement: why has this problem not been solved before?
* solution statement: what specific element of the challenge does your research resolve?
* objective: how will you accomplish this?
* literature review: focus on foundational or methods papers that closely describe your analytical workflow.
* research questions: focus on testable hypothesis, even if you're not a frequentist. 


## Data Sources

Describe your study area, and period of interest. Specify whether training data represents a different location/time period than forecast simulations. Detail the temporal and spatial frequency of your process.

### Published Data Sources
| Name | Source | Description | Access Method | data DOI/url | metadata DOI/URL| details | data citation |
|------|--------|-------------|---------------|--------------|-----------------|---------|---------------|

### Data Access Notes
Many public geospatial data repositories require user authentication to access data. In the methods section, detail which data sources require registraton to access. Link to sign up portals for any listed data sources that require user authentication. In the "How to Reproduce," descripe how to configure automatic access control mechanisms for each data source. 

### Inputs folder
Any direct data download links can be pasted into the "datalinks.txt" file in the inputs folder. Specify which dataset links can be accessed via the datalinks.txt folder. Note: this should only be used for PDIs: if the url changes, it will break the reproducibility of your workflow.

Detail any datasets that are in your inputs data folder. Note this is only for data that is too small/trivial to be published: **no files greater than 10 MB can be stored in repository**. Examples might include spatial polygons that have undergone geometry simplification for API searches, text-based keys mapping variable names to integer values, etc.

## Methods Summary

**Model Framework:** {{ cookiecutter.model_framework }}
Describe steps involved in data preprocessing

## Repository Structure

| Folder/File | Description |
|-------------|-------------|
| notebooks/ | SE1–SE4 notebooks |
| inputs/ | minimal input data required, note most data should be stored on OGC/FAIR compliant databases and accessed from stable URLs |
| processed_data/ | analysis-ready datasets |
| model_data/ | Saved model outputs, model configuration files, predictions|
| figures/ | Figures, tables, graphs, and data-derivatives (e.g. summary statistics) displayed in manuscript text |
| run_reproducibility.py | Reproducibility wrapper |
| Dockerfile | Reproducible container |
| CITATION.cff | Citation metadata, sourced directly from Zenodo |

## How to Reproduce

### Computational requirements
What operating system, processor type, and processor specifications (RAM, cores, etc).

If GPU processing, specify CUDA version.

### Data access configurations
Describe in detail any access control mechanisms that need to be configured for an individual user to access data (e.g. tokens, cookies, certificates, URL customization). Provide links to documentation.

### Run the code
```bash
pip install -r requirements.txt
python run_reproducibility.py
```
## Results

Display key figures in `/figures` folder, with description:

![Example](figures/example.png)

## Citation
All repositories should be published on a platform providing persistent object identifiers (e.g. Zenodo).

DOI: **DOI_PENDING**

## License

{{ cookiecutter.license }}

## Contribution Guidelines
Contributions that improve the quality, clarity, and reproducibility of this project are welcome.
* Open an issue before making major or result-affecting changes.
* Keep pull requests focused and clearly describe what changed and why.
* Follow existing code style and update documentation as needed.
* Do not modify code or data used to reproduce published results without discussion.
* Ensure workflows remain reproducible (environment, dependencies, random seeds).
* Do not commit large or restricted datasets; respect data licenses.
By contributing, you agree that your work will be released under the project’s license.

## Notes:
Focus on graphically rich, interactive elements to communicate your research to diverse stakeholders.
[Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/markdown-cheatsheet) 