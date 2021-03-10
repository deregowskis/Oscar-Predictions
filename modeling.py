import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline

training_data = pd.read_csv('Oscars-TrainingData.csv',sep=';',encoding='latin1')
testing_data = pd.read_csv('Oscars-TestingData.csv',sep=';',encoding='latin1')

def PredictCategory(input_category):
    categories_to_analyse = [[], ['GG-Drama', 'GG-DramaW', 'GG-Comedy', 'GG-ComedyW', 'B-Film',
                                  'CC-Film', 'CC-FilmW', 'CC-Komedia', 'CC-KomediaW', 'PGA'],
                             ['GG-Dir', 'GG-DirW', 'B-Dir', 'CC-Dir', 'CC-DirW', 'DGA'],
                             ['GG-DramaM1', 'GG-DramaM1W', 'GG-ComedyM1', 'GG-ComedyM1W',
                              'B-M1', 'CC-M1', 'CC-M1W', 'SAG-M1'],
                             ['GG-DramaK1', 'GG-DramaK1W', 'GG-ComedyK1', 'GG-ComedyK1W',
                              'B-K1', 'CC-K1', 'CC-K1W', 'SAG-K1'],
                             ['GG-M2', 'GG-M2W', 'B-M2', 'CC-M2', 'CC-M2W', 'SAG-M2'],
                             ['GG-K2', 'GG-K2W', 'B-K2', 'CC-K2', 'CC-K2W', 'SAG-K2'],
                             ['GG-Screen', 'GG-ScreenW', 'B-ScreenO', 'CC-ScreenO', 'CC-ScreenOW', 'WGA-ScreenO'],
                             ['GG-Screen', 'GG-ScreenW', 'B-ScreenA', 'CC-ScreenA', 'CC-ScreenAW', 'WGA-ScreenA'],
                             ['B-Zdj', 'CC-Zdj', 'CC-ZdjW', 'ASC'],
                             ['GG-Music', 'GG-MusicW', 'B-Music', 'CC-Music', 'CC-MusicW'],
                             ['B-Scen', 'CC-Scen', 'CC-ScenW', 'ADG'],
                             ['B-Kost', 'CC-Kost', 'CC-KostW', 'CDG'],
                             ['B-ChF', 'CC-ChF', 'CC-ChFW'],
                             ['B-Efekty', 'CC-Efekty', 'CC-EfektyW'],
                             ['B-Sound', 'ZS'],
                             ['B-Mon','CC-Mon','CC-MonW']
                             ]
    category_to_predict = [[], ['O-Film', ], ['O-Dir'], ['O-M1'], ['O-K1'], ['O-M2'], ['O-K2'], ['O-ScreenO'],
                           ['O-ScreenA'],
                           ['O-Zdj'], ['O-Muzyka'], ['O-Scen'], ['O-Kost'], ['O-ChF'], ['O-Efekty'], ['O-Sound'],['O-Mont']]
    pipe = Pipeline(
        [
            ('linear-model', LinearRegression())
        ]
    )
    pipe.fit(training_data[categories_to_analyse[input_category]],
                    training_data[category_to_predict[input_category]])
    pipe_prediction = pipe.predict(testing_data[categories_to_analyse[input_category]])
    predictions_df = pd.DataFrame(pipe_prediction)
    testing_data[category_to_predict[input_category]] = predictions_df
    if input_category == 1:
        number = 10
    else:
        number = 5
    sorted = testing_data.sort_values(by=category_to_predict[input_category], ascending=False).head(number)
    list_of_nominees = sorted.iloc[:, 0].values
    print("Predictions for this category:")
    for element in list_of_nominees:
        print(element)

def main():
    print("Choose category to predict:")
    print("1 - Best Picture")
    print("2 - Best Directing")
    print("3 - Best Leading Actor")
    print("4 - Best Leading Actress")
    print("5 - Best Supporting Actor")
    print("6 - Best Supporting Actress")
    print("7 - Best Original Screenplay")
    print("8 - Best Adapted Screenplay")
    print("9 - Best Cinematography")
    print("10 - Best Original Score")
    print("11 - Best Production Design")
    print("12 - Best Costumes")
    print("13 - Best Make-up and Hairstyling")
    print("14 - Best Visual Effects")
    print("15 - Best Sound Mixing")
    print("16 - Best Film Editing")
    print()
    input_category = input("Type a number (1-16): ")
    while (input_category != "1" and
           input_category != "2" and
           input_category != "3" and
           input_category != "4" and
           input_category != "5" and
           input_category != "6" and
           input_category != "7" and
           input_category != "8" and
           input_category != "9" and
           input_category != "10" and
           input_category != "11" and
           input_category != "12" and
           input_category != "13" and
           input_category != "14" and
           input_category != "15" and
           input_category != "16"):
        input_category = input("Incorrect input. Type a number (1-16): ")
    input_category = int(input_category)
    print()
    PredictCategory(input_category)
    print()
    input_another = input("Would you like to predict another category? (y/n): ")
    while (input_another != "y" and input_another != "n"):
        input_another = input("Incorrect input. Would you like to predict another category? (y/n): ")
    if(input_another=="y"):
        main()
    if(input_another=="n"):
        input_last = input("Type ENTER to quit.")

if __name__=="__main__":
    main()