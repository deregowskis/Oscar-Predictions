Oscar nominations are usual the resultant of nominations for less prestigious awards (e.g. Golden Globes, BAFTA, Critics Choice, industry awards). A standard linear 
regression model of machine learning is used to predict 2021 Academy Awards nominations based on 4 previous years' nominations.

***Oscars-TrainingData.csv*** contains all data about 5 last years, including nominations for Oscars, Golden Globes and so on (234 movies x 100 categories). For every movie there is a dedicated number (mostly 0 or 1, sometimes 2) that indicates how many nominations did the film achieve in a given category. This is a training data for the model.

***Oscars-TestingData.csv*** contains the same data about 2021 movies except for Oscars nominations (61 movies x 84 categories).

All the collected data is based on [filmweb.pl](https://www.filmweb.pl/awards).

***modeling.py*** is an executable file which enables a user to pick a category and see a predicted nominees.

***preview.png*** is a view from Python Console while executing modeling.py.

After Oscars nominations' announcement (March 15th, 2021) I'll upload a file summarizing accuracy of my model.
