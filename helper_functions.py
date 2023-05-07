import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SeabornFig2Grid as sfg
import pandas as pd
import glob
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils.vis_utils import plot_model

data_dir = r"./Data/"


def data_load_and_preprocessing():
    # data import
    clinical_data = pd.read_csv(os.path.join(data_dir, "ADRCClinicalData.csv"))
    subjects = pd.read_csv(os.path.join(data_dir, "subjects.csv"))
    freesurfers = pd.read_csv(os.path.join(data_dir, "FreeSurfers.csv"))

    subjects_list = subjects["Subject"]

    # define cognitively normal participants as participants with min and max cdr score of zero
    list_cog_normal = []
    for sub in subjects_list:
        y = clinical_data[clinical_data["Subject"] == sub]["cdr"].values
        if np.all(y == 0):
            list_cog_normal.append(sub)

    # freesurfer data columns
    list_of_plot = ['IntraCranialVol',
                    'lhCortexVol',
                    'rhCortexVol',
                    'CortexVol',
                    'SubCortGrayVol',
                    'TotalGrayVol',
                    'SupraTentorialVol',
                    'lhCorticalWhiteMatterVol',
                    'rhCorticalWhiteMatterVol',
                    'CorticalWhiteMatterVol']

    # combine data across tables

    subjects_combined = []
    # add gender column
    subjects_combined = pd.DataFrame(subjects.set_index("Subject")["M/F"])

    # add age at entry column
    age_entry_list = []
    for sub in subjects["Subject"]:
        age_entry_list.append(clinical_data[clinical_data["Subject"] == sub]["ageAtEntry"].iloc[0])
    age_entry_list = pd.DataFrame(data=age_entry_list, index=subjects["Subject"])
    subjects_combined["ageAtEntry"] = age_entry_list

    # merge gender and age at entry data to freesurfer data
    new_free_surfer = freesurfers.set_index("Subject").merge(subjects_combined, on="Subject")


    # calculate age at freesurfer measurement and add it as a column
    new_free_surfer["days_since_first"] = [int(i[-4:]) for i in new_free_surfer["FS_FSDATA ID"]]
    new_free_surfer["years_since_first"] = new_free_surfer["days_since_first"] / 365
    new_free_surfer["real_age"] = new_free_surfer["years_since_first"] + new_free_surfer["ageAtEntry"]

    merged_subjects_data = new_free_surfer

    # add column for AD diagnosis
    merged_subjects_data["AD_diag"] = [sub not in list_cog_normal for sub in new_free_surfer.index]

    # separate participants into age ranges
    bins = [40, 50, 60, 70, 80, 90, 100]
    labels = ["40-50",
              "50-60",
              "60-70",
              "70-80",
              "80-90",
              "90-100"
              ]
    merged_subjects_data["age_ranges"] = pd.cut(merged_subjects_data["real_age"], bins, labels=labels,
                                                include_lowest=True)

    # separate male and female
    male_data = merged_subjects_data.loc[merged_subjects_data["M/F"] == "M"]
    female_data = merged_subjects_data.loc[merged_subjects_data["M/F"] == "F"]

    return list_of_plot, merged_subjects_data, male_data, female_data


# violin plots
def violin_plots(data):
    list_of_plot, merged_subjects_data, male_data, female_data = data
    for param in list_of_plot:
        f, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax[0].set_title("Male %s" % param)
        sns.violinplot(x="age_ranges", y=param, data=male_data, hue="AD_diag", split=False,
                       ax=ax[0])  # , inner="quartile")
        # plt.show()

        # plt.subplots(figsize=(10, 5))
        ax[1].set_title("Female %s" % param)
        sns.violinplot(x="age_ranges", y=param, data=female_data, hue="AD_diag", split=False,
                       ax=ax[1])  # , inner="quartile")
        plt.show()


def scatter_plots(data):
    list_of_plot, merged_subjects_data, male_data, female_data = data

    # 2D scatter plota
    for param in list_of_plot:
        #     f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
        g0 = sns.jointplot(data=male_data, x="real_age", y=param, hue="AD_diag", alpha=0.5)
        g1 = sns.jointplot(data=female_data, x="real_age", y=param, hue="AD_diag", alpha=0.5)
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 2)
        mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
        mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
        gs.tight_layout(fig)
        plt.title(param)
        plt.show()


# beginning of machine learning

def run_machine_learning(data):
    list_of_plot, merged_subjects_data, male_data, female_data = data

    input_columns = ['IntraCranialVol',
                     'lhCortexVol',
                     'rhCortexVol',
                     'CortexVol',
                     'SubCortGrayVol',
                     'TotalGrayVol',
                     'SupraTentorialVol',
                     'lhCorticalWhiteMatterVol',
                     'rhCorticalWhiteMatterVol',
                     'CorticalWhiteMatterVol',
                     'real_age',
                     'gender_num']

    # add gender as a numerical value
    merged_subjects_data['gender_num'] = pd.DataFrame(merged_subjects_data["M/F"] == "M").astype(int)

    # define input and output
    data_y = merged_subjects_data["AD_diag"].astype(int)
    data_X = sklearn.preprocessing.normalize(merged_subjects_data[input_columns], axis=0, norm="max") # normalize input

    # separate train and test data
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.20, random_state=42)

    # define nn model
    model = Sequential()

    model.add(Dense(30, input_dim=12, activation='relu'))

    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(15, activation='relu'))

    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=200, batch_size=32)
    return model, X_test, y_test


def eval_machine_learning(data_ml):
    model, X_test, y_test = data_ml
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))


def plot_ml_model(data_ml):
    model, _, _ = data_ml
    plt.subplots()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    plt.show()


if __name__ == '__main__':
    data = data_load_and_preprocessing()
    violin_plots(data)
    scatter_plots(data)
    run_machine_learning(data)
