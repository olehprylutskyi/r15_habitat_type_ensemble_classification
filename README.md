# Ensemble classification of chalk outcrops in Ukraine using Random Forest with bagging, similarity in satellite embedding space, and vegetation releves

## Purpose
The range of vegetation communities on chalk outcrops spans from the southwestern spurs of the Central European Upland and the Donets Ridge in the west to the spurs of the Volga Upland in the east and hardly goes beyond the boundaries of the Don River basin, with the exception of the eastern boundary, where chalky rocks are exposed along the right bank of the Volga River  (Kuzemko et al, 2022). This habit type is listed in Resolution 4 of the Bern Convention as E1.13 - Continental dry rocky steppic grasslands and dwarf scrub on chalk outcrops and is thus protected at the European level. According to the EUNIS habitat classification system, this habitat type corresponds with type [R15 - Continental dry rocky steppic grassland and dwarf scrub on chalk outcrops](https://biodiversity.europa.eu/habitats_eunis_revised/EUNISrev_R15). Under active combat during Russian invasion in Ukraine, most of this habitats suffered from explosive ordnance and tranches, posing as a case of ecocide.

## Methods
Using Alpha Earth Foundation data and expert-confirmed locations of chalk outcrops retrieved from curated vergetation releves, this script performs ensemble classification of this habitat type in Eastern Ukraine. With only eight presence points and no confirmed absences, random background ("pseudo-absence") points were used to represent other habitat types, following the presence-only framework of Barbet-Massin et al. (2012). To reduce
sensitivity to the arbitrary placement of any single background sample, two independent strategies were run and combined: (A) a bagged Random Forest trained repeatedly against freshly drawn background samples, and (B) an
unsupervised similarity measure requiring no background sample at all. Agreement between the two was used as a data-independent confidence indicator.

## Data sources

Earth observation data: [Google Satellite Embedding V1 (annual)](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)

Ground truth: vegetation releves by Anna Kuzemko, [2017](https://code.earthengine.google.com/?asset=projects/ee-olegpril12/assets/habitats/releves_r15)

Area of interest: grid cells from Kuzemko et al. ([2021](https://code.earthengine.google.com/?asset=projects/ee-olegpril12/assets/habitats/grid_ukraine_wgs84_grasslandAtlas_2021)) Atlas of the
Grassland Habitats of Ukraine (ISBN 978-617-7849-93-2)

![Fig. 1. R15 habitat type](https://github.com/olehprylutskyi/r15_habitat_type_ensemble_classification/blob/main/fig_repo.jpeg)

A – Classification agreement between supervised (Random Forest with bagging) and unsupervised (cosine similarity in embedding space). B – Probability of a pixel belongs to R15 habitat type, obtained by Random Forest supervised classification with bagging. C – Probability of a pixel belongs to R15 habitat type, obtained by Alpha Earth dot product cosine distance from confirmed points (centroid similarity). C – Google Hybrid basemap (chalk outcrops are visible as bright spots along the river).

## Requirements
- Google Earth Engine account

## Author
[Oleh Prylutskyi](https://orcid.org/0000-0001-5730-517X)

## Bibliographic citation
Anna Kuzemko, Taras Kazantsev, Oleh Prylutsky, Andrii Tupikov & Olesandr Khodosovtsev. APPLYING THE HABITAT APPROACH TO THE QUALIFICATION OF ECOCIDE: CASE STUDIES FROM THE RUSSIAN-UKRAINIAN WAR (in press)