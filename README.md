The deployed version of this shiny app can be found at: 

https://lsluijterman.shinyapps.io/crps_calculator/

There, you can upload csv files with observations and predictions and get the following metrics:

- The average CRPS along with a bootstrap 90% confidence interval. 
- The predicted-to-observed geometric mean ratio along with a parametric 90% confidence interval.
- The skill score, which is given by 1 - CRPS(model) / CRPS(naive model), where the naive model always predicts the sample median. 

Additionally, visulisations of the PDFs and CDFs are given for a visual evaluation.

Note that all these metrics are calculated for a model that estimates the marginal distribution. This means that we do not have individual predictions for each observations but rather a large set of predictions that aim to mimic the distribution of the observations. It is therefor advised to use a large set of predictions as this gives a more accurate estimate of the predicted distribution.