import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter("ignore")

training_data = pd.read_csv('OscarsWinners-TrainingData.csv', sep=';', encoding='latin1')
testing_data = pd.read_csv('OscarsWinners-TestingData.csv', sep=';', encoding='latin1')
testing_data = testing_data.fillna(0)
training_data = training_data.fillna(0)


def PredictCategory(input_category):
    all_categories_to_analyse = ['Year', 'GG-Drama', 'GG-DramaW', 'GG-Comedy',
                                 'GG-ComedyW', 'GG-DramaM1', 'GG-DramaM1W', 'GG-ComedyM1',
                                 'GG-ComedyM1W', 'GG-DramaK1', 'GG-DramaK1W', 'GG-ComedyK1',
                                 'GG-ComedyK1W', 'GG-M2', 'GG-M2W', 'GG-K2', 'GG-K2W', 'GG-Dir',
                                 'GG-DirW', 'GG-Screen', 'GG-ScreenW', 'GG-Music', 'GG-MusicW',
                                 'B-Film', 'B-FilmW', 'B-M1', 'B-M1W', 'B-K1', 'B-K1W', 'B-M2',
                                 'B-M2W', 'B-K2', 'B-K2W', 'B-Dir', 'B-DirW', 'B-ScreenO',
                                 'B-ScreenOW', 'B-ScreenA', 'B-ScreenAW', 'B-Zdj', 'B-ZdjW',
                                 'B-Music', 'B-MusicW', 'B-Scen', 'B-ScenW', 'B-Kost', 'B-KostW',
                                 'B-Mon', 'B-MonW', 'B-ChF', 'B-ChFW', 'B-Efekty', 'B-EfektyW',
                                 'B-Sound', 'B-SoundW', 'CC-Film', 'CC-FilmW', 'CC-Komedia',
                                 'CC-KomediaW', 'CC-K1', 'CC-K1W', 'CC-M1', 'CC-M1W', 'CC-K2',
                                 'CC-K2W', 'CC-M2', 'CC-M2W', 'CC-Dir', 'CC-DirW', 'CC-ScreenA',
                                 'CC-ScreenAW', 'CC-ScreenO', 'CC-ScreenOW', 'CC-Music',
                                 'CC-MusicW', 'CC-Scen', 'CC-ScenW', 'CC-Kost', 'CC-KostW',
                                 'CC-Zdj', 'CC-ZdjW', 'CC-Mon', 'CC-MonW', 'CC-Efekty',
                                 'CC-EfektyW', 'CC-ChF', 'CC-ChFW', 'PGA', 'PGAW', 'DGA', 'DGAW',
                                 'SAG-team', 'SAG-teamW', 'SAG-M1', 'SAG-M1W', 'SAG-K1', 'SAG-K1W',
                                 'SAG-M2', 'SAG-M2W', 'SAG-K2', 'SAG-K2W', 'WGA-ScreenO',
                                 'WGA-ScreenOW', 'WGA-ScreenA', 'WGA-ScreenAW', 'ADG', 'ADGW',
                                 'CDG', 'CDGW', 'ZS', 'ZSW', 'ASC', 'ASCW', 'ACE', 'ACEW', 'O-Film',
                                 'O-Dir', 'O-M1', 'O-K1',
                                 'O-M2', 'O-K2', 'O-ScreenO',
                                 'O-ScreenA', 'O-Zdj', 'O-Muzyka',
                                 'O-Scen', 'O-Kost', 'O-ChF',
                                 'O-Efekty', 'O-Mont', 'O-Sound']
    categories_to_analyse = [[], ['Year', 'GG-Drama', 'GG-DramaW', 'GG-Comedy', 'GG-ComedyW', 'B-Film', 'B-FilmW',
                                  'CC-Film', 'CC-FilmW', 'CC-Komedia', 'CC-KomediaW', 'PGA', 'PGAW', 'O-Film'],
                             ['Year', 'GG-Dir', 'GG-DirW', 'B-Dir', 'B-DirW', 'CC-Dir', 'CC-DirW', 'DGA', 'DGAW',
                              'O-Dir'],
                             ['Year', 'GG-DramaM1', 'GG-DramaM1W', 'GG-ComedyM1', 'GG-ComedyM1W',
                              'B-M1', 'B-M1W', 'CC-M1', 'CC-M1W', 'SAG-M1', 'SAG-M1W', 'O-M1'],
                             ['Year', 'GG-DramaK1', 'GG-DramaK1W', 'GG-ComedyK1', 'GG-ComedyK1W',
                              'B-K1', 'B-K1W', 'CC-K1', 'CC-K1W', 'SAG-K1', 'SAG-K1W', 'O-K1'],
                             ['Year', 'GG-M2', 'GG-M2W', 'B-M2', 'B-M2W', 'CC-M2', 'CC-M2W', 'SAG-M2', 'SAG-M2W',
                              'O-M2'],
                             ['Year', 'GG-K2', 'GG-K2W', 'B-K2', 'B-K2W', 'CC-K2', 'CC-K2W', 'SAG-K2', 'SAG-K2W',
                              'O-K2'],
                             ['Year', 'GG-Screen', 'GG-ScreenW', 'B-ScreenO', 'B-ScreenOW', 'CC-ScreenO', 'CC-ScreenOW',
                              'WGA-ScreenO', 'WGA-ScreenOW', 'O-ScreenO'],
                             ['Year', 'GG-Screen', 'GG-ScreenW', 'B-ScreenA', 'B-ScreenAW', 'CC-ScreenA', 'CC-ScreenAW',
                              'WGA-ScreenA', 'WGA-ScreenAW', 'O-ScreenA'],
                             ['Year', 'B-Zdj', 'CC-Zdj', 'CC-ZdjW', 'ASC', 'ASCW', 'O-Zdj'],
                             ['Year', 'GG-Music', 'GG-MusicW', 'B-Music', 'B-MusicW', 'CC-Music', 'CC-MusicW',
                            'O-Muzyka'],
                             ['Year', 'B-Scen', 'B-ScenW', 'CC-Scen', 'CC-ScenW', 'ADG', 'ADGW', 'O-Scen'],
                             ['Year', 'B-Kost', 'B-KostW', 'CC-Kost', 'CC-KostW', 'CDG', 'CDGW', 'O-Kost'],
                             ['Year', 'B-ChF', 'B-ChFW', 'CC-ChF', 'CC-ChFW', 'O-ChF'],
                             ['Year', 'B-Efekty', 'B-EfektyW', 'CC-Efekty', 'CC-EfektyW', 'O-Efekty'],
                             ['Year', 'B-Sound', 'B-SoundW', 'ZS', 'ZSW', 'O-Sound'],
                             ['Year', 'B-Mon', 'B-MonW', 'CC-Mon', 'CC-MonW', 'ACE', 'ACEW', 'O-Mont']
                             ]
    category_to_use = [[], ['O-Film'], ['O-Dir'], ['O-M1'], ['O-K1'], ['O-M2'], ['O-K2'], ['O-ScreenO'],
                       ['O-ScreenA'],
                       ['O-Zdj'], ['O-Muzyka'], ['O-Scen'], ['O-Kost'], ['O-ChF'], ['O-Efekty'], ['O-Sound'],
                       ['O-Mont']]
    category_to_predict = [[], ['O-FilmW'], ['O-DirW'], ['O-M1W'], ['O-K1W'], ['O-M2W'], ['O-K2W'], ['O-ScreenOW'],
                           ['O-ScreenAW'],
                           ['O-ZdjW'], ['O-MuzykaW'], ['O-ScenW'], ['O-KostW'], ['O-ChFW'], ['O-EfektyW'], ['O-SoundW'],
                           ['O-MontW']]

    model_logreg = LogisticRegression(penalty='l1', solver='saga')
    model_logreg.fit(
        training_data[training_data[category_to_use[input_category][0]] > 0][categories_to_analyse[input_category]],
        training_data[training_data[category_to_use[input_category][0]] > 0][
            category_to_predict[input_category]].values.ravel())
    model_logreg_prediction = model_logreg.predict_proba(
        testing_data[testing_data[category_to_use[input_category][0]] > 0][categories_to_analyse[input_category]])
    model_logreg_prediction_df = pd.DataFrame()
    model_logreg_prediction_df['Title'] = testing_data[testing_data[category_to_use[input_category][0]] > 0][
        'Title'].values.ravel()
    model_logreg_prediction_df['Chances (%)'] = model_logreg_prediction[:, 1]
    score_sum = model_logreg_prediction_df['Chances (%)'].values.ravel().sum()
    model_logreg_prediction_df['Chances (%)'] = model_logreg_prediction_df['Chances (%)'] / score_sum * 100
    print("Predictions for this category (logistic regression model):")
    print(model_logreg_prediction_df.sort_values(by='Chances (%)', ascending=False))

    model_xgb = XGBClassifier(random_state=1,
                              learning_rate=0.001,
                              booster='gbtree',
                              importance_type='cover',
                              nround=1000,
                              max_depth=4,
                              verbosity=0,
                              use_label_encoder=False)
    model_xgb.fit(
        training_data[training_data[category_to_use[input_category][0]] > 0][categories_to_analyse[input_category]],
        training_data[training_data[category_to_use[input_category][0]] > 0][
            category_to_predict[input_category]].values.ravel())
    model_xgb_prediction = model_xgb.predict_proba(
        testing_data[testing_data[category_to_use[input_category][0]] > 0][categories_to_analyse[input_category]])
    model_xgb_prediction_df = pd.DataFrame()
    model_xgb_prediction_df['Title'] = testing_data[testing_data[category_to_use[input_category][0]] > 0][
        'Title'].values.ravel()
    model_xgb_prediction_df['Chances (%)'] = model_xgb_prediction[:, 1]
    score_sum = model_xgb_prediction_df['Chances (%)'].values.ravel().sum()
    model_xgb_prediction_df['Chances (%)'] = model_xgb_prediction_df['Chances (%)'] / score_sum * 100
    print("Predictions for this category (XGBClassifier model):")
    print(model_xgb_prediction_df.sort_values(by='Chances (%)', ascending=False))

    model_svc = SVC(probability=True)
    model_svc.fit(
        training_data[training_data[category_to_use[input_category][0]] > 0][categories_to_analyse[input_category]],
        training_data[training_data[category_to_use[input_category][0]] > 0][
            category_to_predict[input_category]].values.ravel())
    model_svc_prediction = model_svc.predict_proba(
        testing_data[testing_data[category_to_use[input_category][0]] > 0][categories_to_analyse[input_category]])
    model_svc_prediction_df = pd.DataFrame()
    model_svc_prediction_df['Title'] = testing_data[testing_data[category_to_use[input_category][0]] > 0][
        'Title'].values.ravel()
    model_svc_prediction_df['Chances (%)'] = model_svc_prediction[:, 1]
    score_sum = model_svc_prediction_df['Chances (%)'].values.ravel().sum()
    model_svc_prediction_df['Chances (%)'] = model_svc_prediction_df['Chances (%)'] / score_sum * 100
    print("Predictions for this category (SVClassifier model):")
    print(model_svc_prediction_df.sort_values(by='Chances (%)', ascending=False))

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
    print("15 - Best Sound")
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
    if (input_another == "y"):
        main()
    if (input_another == "n"):
        input_last = input("Type ENTER to quit.")


if __name__ == "__main__":
    main()
