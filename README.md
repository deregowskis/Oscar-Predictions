Oscar nominations and winners are usual the resultant of nominations and winners of less prestigious awards (e.g. Golden Globes, BAFTA, Critics Choice, industry awards). Supervised machine learning models are used to predict 2021 Academy Awards nominations and winners based on 5 previous years' nominations.

***Nominations - summary.pdf*** contains a summarized report about predictions of nominations.

***Winners - summary.pdf*** contains a summarized report about predictions of winners.

***Oscars-TrainingData.csv*** contains all data about 5 last years, including nominations for Oscars, Golden Globes and so on (234 movies x 100 categories). For every movie there is a dedicated number (mostly 0 or 1, sometimes 2) that indicates how many nominations did the film achieve in a given category. This is a training data for the model.

***Oscars-TestingData.csv*** contains the same data about 2021 movies except for Oscars nominations (61 movies x 84 categories).

***OscarsWinners-TrainingData.csv*** contains all data from *Oscars-TrainingData* and additionaly info about winners of particular Oscar categories.

***OscarsWinners-TrainingData.csv*** contains all data from *Oscars-TestingData* and additionaly info about nominations for particular Oscar categories.

All the collected data is based on [filmweb.pl](https://www.filmweb.pl/awards).

***OscarsNominations.py*** is an executable file which enables a user to pick a category and see a predicted nominees.

***OscarsWinners.py*** is an executable file which enables a user to pick a category and see a predicted winner.

***preview.png*** and ***preview2.png*** are views from Python Console while executing OscarsNominations.py.
