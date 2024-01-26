# ExoNet: A Machine Learning Classifier for Vetting Planet Candidates in TESS Data

![Transit Animation](docs/transit-method-single-planet.mp4)

## Background
ExoNet is a repository created for the testing of my thesis for bachelor degree in computer science at university of Naples Parthenope.

## Abstract

In this thesis, we explore the application of Machine Learning (ML) techniques to the classification of exoplanets, focusing on data from NASA's TESS telescope. The TESS space telescope, currently in orbit, detects exoplanets by using the transit method, i.e., by measuring the relative decrease in stellar luminosity caused by a planet crossing the stellar disk along the line of sight.

When a transiting event is detected with a Signal to Noise ratio of at least seven, it is classified as a Threshold Crossing Event (TCE) and analyzed by data reduction centers, such as the Science Processing Operations Center (SPOC), or by automated pipelines like the Quick-Look Pipeline (QLP). Based on the products of this analysis, the most interesting candidates are reported for follow-up observations, conducted manually by experts, and validated.

The goal of this work is to automate the identification process of the most promising candidate planets for which follow-ups are to be scheduled, using ML techniques. However, the scarcity of labeled TCEs in TESS data (6,977 total observations, including 5,422 planets and 1,555 false planets) and the imbalanced nature of the dataset requires the inclusion of additional data from the Kepler telescope, allowing us to obtain a merged dataset with 13,798 total observations, including 8,396 planets and 5,402 false planets.

Based on this dataset, various ML algorithms have been explored, ranging from supervised approaches to unsupervised ones. Among the supervised approaches that we tested, the Random Forest model achieves the best performance, with an accuracy of 83.69%, and it is also easily interpretable. Among the unsupervised approaches, the Self-Organizing Maps are particularly useful for the visualization of the data. We also explored their use for binary classification and obtained an accuracy of 77.57%, which, although lower than the one achieved by the Random Forest, is very promising and suggests further opportunities for study based on non-linear maps.

## Installazione

1. Clona il repository
   ```bash
   git clone https://github.com/Attilio-Di-Vicino/ExoNet.git
   ```

2. Installa i requirements
   ```bash
   conda create --name exonet --file requirements.txt
   ```

## Licenza

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](LICENSE).

## Citation
if you want cite me
    ```bash
    @misc{github.exonet,
        title = {{ExoNet}},
        howpublished = {\url{https://github.com/Attilio-Di-Vicino/ExoNet}},
        author = {Attilio Di Vicino},
        year = {2023},
        description = {ExoNet is a repository created for the testing of my thesis entitled "A Machine Learning Classifier for Vetting Planet Candidates in TESS Data".}
    }
    ```

## Contact 

Attilio Di Vicino: attilio.divicino001@studenti.uniparthenope.it