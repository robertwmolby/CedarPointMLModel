# Cedar Point Attendance & Roller Coaster Recommendation -- Python Project

This repository contains a demonstration project built to explore,
model, and serve predictions around amusement-park attendance and
roller-coaster recommendations.\
It includes data scraping, machine-learning models,
recommender logic, and Flask APIs deployed in Docker containers
onto Amazon EKS.

Although this is currently a single-user project, it has been
developed using a generalized Git-Flow workflow---feature branches,
develop/staging integration, and tagged releases---so that the project
can scale to multi-developer collaboration and CI/CD pipelines if
needed.

The project is organized into three major components.

------------------------------------------------------------------------

## 1. Clustering & Anomaly Detection on Cedar Point Attendance

This module performs exploratory data analysis, clustering, and
anomaly detection on historical Cedar Point crowd data.\
Data is collected from publicly available queue-time sources and
enriched with additional metadata such as weather and event details.

Features include:

-   Data scraping and normalization
-   Unsupervised clustering (K-Means, DBSCAN, hierarchical methods)
-   Outlier and anomaly identification
-   Visualization utilities
-   Seasonality and pattern discovery across the Cedar Point operating
    calendar

------------------------------------------------------------------------

## 2. Attendance Prediction Models with Flask API

This module builds and evaluates a variety of supervised
machine-learning models designed to predict Cedar Point crowd levels.
Models incorporate:

-   Holiday and school calendars
-   Park event schedules
-   Weather forecasts and historical weather conditions
-   Day-of-week and seasonal trends
-   Engineered features from historical attendance patterns

Multiple model types were developed, including:

-   Linear and polynomial regression
-   Random Forest, XGBoost, LightGBM
-   Neural networks

A training routine compares all models, selects the best-performing one,
and then exports it for deployment.
The resulting logic is written as standalone, pipeline-ready Python
modules.

The resultant and selected model has been encapsulated along with a Flask API and deployed as a Docker image in AWS.  It can currently be found here: http://a90649d1769f742f796d36ccdfe62156-331747868.us-east-2.elb.amazonaws.com/docs  

------------------------------------------------------------------------

## 3. Roller Coaster Recommendation Engine (with Flask API)

The third module provides a roller-coaster recommendation system
derived from global roller-coaster metadata and user rating data.

It supports:

-   Processing worldwide roller-coaster information
-   Integrating user ratings
-   Generating personalized recommendations
-   Providing coaster similarity lookups
-   Training and exporting recommendation models

This functionality has been exposed as an API built with Flask which is deployed to AWS EKS in the form of Docker images.  Details of the endpoints created can currently be found here:  http://ac3a45cc0862c4debaeed73d6650680d-1292001656.us-east-2.elb.amazonaws.com/docs

### Flask API Layer

All recommendation features are exposed through a lightweight Flask
REST API, packaged into Docker images, and deployed to AWS EKS.

These APIs are consumed by a separate Spring Boot application, which
provides orchestration and external interface layers.

-   Spring Boot Swagger UI:\
    http://a3edcc49fa0874231a028402c7ba9ebf-41972715.us-east-2.elb.amazonaws.com/swagger-ui/index.html

-   Spring Boot GitHub Repository:\
    https://github.com/robertwmolby/RollerCoasterRecommender.git

------------------------------------------------------------------------

## Technologies Used

-   Python 3.x
-   Flask for REST endpoints
-   scikit-learn, XGBoost, LightGBM, TensorFlow/PyTorch
-   pandas, NumPy
-   BeautifulSoup, Requests
-   Docker and Amazon EKS for deployment
-   Spring Boot for external application integration

------------------------------------------------------------------------

## Deployment

The recommendation module is currently a deployable unit while the others exist as notebooks.  It has been deployed to Kubernetes at Amazon Web Services (AWS EKS) as docker images.

Kubernetes artifacts include:

-   Deployments
-   Services (LoadBalancer)
-   Secrets
-   Health probes and resource definitions


