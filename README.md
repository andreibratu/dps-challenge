# DSP Challenge

**Video walktrough available here: https://www.loom.com/share/2cad937816304500b2ea4307dd9ffcf7**

## Bonus points

- [x] Custom XGB model used (but not performing as better)

- [x] Video walkthrough provided (longer than I wanted, feel free to skip or 2x)

- [ ] Custom ML task

## Analysis part

- `auto-mpg.csv` represents the initial dataset

- `analysis.ipynb` and `analysis2.ipynb` are my attempts at feature engineering. Tried adding manufacturer column and extended `origin` to exact country of origin

- Conclusion is that the initial dataset is the best approach.

- Had the feature engineering revealed better features than the 9 initial ones, I would have probably used default values in the prediction container to make up for the lack of them in the test input.

- `best.csv` is the inital dataset on which OHE columns have been added and all features have standardised (including OHE columns, in accordance with the Tensorflow tutorial)

- Set random_states and seeds to a value of 42 to get reproducible results

## Training

- Files of interest `train.py`, `requirements.txt`, `Dockerfile.train`

- Attempted bayesian hyperparameter search, gave up due to long training times and it being an unasked complication

- 1.995 mean MAE on validation

- Output model is under `model.pkl`. Use `pickle` to load it

- When training on Vertex, the output is stored on GCS

## Deployment/ Prediction

- Files of interest `deploy.py`, `requirements.deploy.txt`, `Dockerfile.deploy`

- Uses FastAPI as HTTP server

- Loads the model from GCS or as backup from local file

- See `example.json` for an example of correct JSON body for a predict request
