"""Exploratory Data Analysis on WSDM Dataset with visualization.

Author: DHSong
Last Modified At: 2020.07.05

Exploratory Data Analysis on WSDM Dataset with visualization.
"""

import os
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

class EDAWorker:
    """Worker for EDA.

    Worker for EDA.

    Attributes:
        train_raw: pandas Dataframe for Train Dataset(train.csv).
        test_raw: pandas Dataframe for Train Dataset(test.csv).
        sample_submission_raw: pandas Dataframe for Submission Dataset(sample_submission.csv).
        songs_raw: pandas Dataframe for Song Dataset(songs.csv).
        members_raw: pandas Dataframe for Member Dataset(members.csv).
        song_extra_info_raw: pandas Dataframe for Additional Song Dataset(song_extra_info.csv).
    """    

    def __init__(self, data_dirname='./data', font_path='./static/fonts/D2Coding.ttc'):
        """Inits Dataframe for data in data directory."""

        self._matplotlib_setting(font_path)

        self.train_raw = pd.read_csv(os.path.join(data_dirname, 'train.csv'))
        self.test_raw = pd.read_csv(os.path.join(data_dirname, 'test.csv'))
        self.sample_submission_raw = pd.read_csv(os.path.join(data_dirname, 'sample_submission.csv'))
        self.songs_raw = pd.read_csv(os.path.join(data_dirname, 'songs.csv'))
        self.members_raw = pd.read_csv(os.path.join(data_dirname, 'members.csv'))
        self.song_extra_info_raw = pd.read_csv(os.path.join(data_dirname, 'song_extra_info.csv'))

    def _matplotlib_setting(self, font_path):
        """set matplotlib fonts and style."""
        font_family = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = 14

        plt.style.use('seaborn-darkgrid')

    def barplot_train_column_by_target(self, column, horizontal=True):
        """Draw barplot.
        
        Draw barplot about columns in train dataset. Visualize distributions of column data based on target value.

        Arags:
            column: str type. Which column in train dataset to be plotted.
            horizontal: bool type. wheter the plot is horizontal or not.

        Return:

        """

        assert column in self.train_raw.columns

        plt.figure(figsize=(16, 9))
        if horizontal:
            sns.countplot(y=column, hue='target', data=self.train_raw, order=self.train_raw[column].value_counts().index)
        else:
            sns.countplot(x=column, hue='target', data=self.train_raw, order=self.train_raw[column].value_counts().index)
        plt.title('{} Distribution by target'.format(column))
        plt.legend(loc='upper right')
        plt.savefig('./figures/barplot_train_column_by_target-{}'.format(column))

    def barplot_members_column(self, column, horizontal=True):
        """Draw barplot.
        
        Draw barplot about columns in members dataset. Visualize distributions of column data.

        Arags:
            column: str type. Which column in members dataset to be plotted.
            horizontal: bool type. wheter the plot is horizontal or not.

        Return:

        """

        assert column in self.members_raw.columns

        plt.figure(figsize=(16, 9))
        if horizontal:
            sns.countplot(y=column, data=self.members_raw, order=self.members_raw[column].value_counts().index)
        else:
            sns.countplot(x=column, data=self.members_raw, order=self.members_raw[column].value_counts().index)
        plt.title('{} Distribution'.format(column))
        plt.savefig('./figures/barplot_members_column-{}'.format(column))

    def barplot_members_column_by_target(self, column, horizontal=True):
        """Draw barplot.
        
        Draw barplot about columns in members dataset. Visualize distributions of column data based on traget value.

        Arags:
            column: str type. Which column in members dataset to be plotted.
            horizontal: bool type. wheter the plot is horizontal or not.

        Return:

        """

        assert column in self.members_raw.columns

        members_train = pd.merge(left=self.members_raw, right=self.train_raw, how='inner', on='msno')
        plt.figure(figsize=(16, 9))
        if horizontal:
            sns.countplot(y=column, hue='target', data=members_train, order=self.members_raw[column].value_counts().index)
        else:
            sns.countplot(x=column, hue='target', data=members_train, order=self.members_raw[column].value_counts().index)
        plt.title('{} Distribution by target'.format(column))
        plt.legend(loc='upper right')
        plt.savefig('./figures/barplot_members_column_by_target-{}'.format(column))

    def kdeplot_members_column(self, column):
        """Draw kdeplot.
        
        Draw kdeplot about columns in members dataset. Visualize distributions of column data.

        Arags:
            column: str type. Which column in members dataset to be plotted.

        Return:

        """

        assert column in self.members_raw.columns

        plt.figure(figsize=(16, 9))
        sns.kdeplot(self.members_raw[column], shade=True)
        plt.title('{} Distribution by target'.format(column))
        plt.savefig('./figures/kdeplot_members_column-{}'.format(column))

    def kdeplot_members_column_by_target(self, column):
        """Draw kdeplot.
        
        Draw kdeplot about columns in members dataset. Visualize distributions of column data based on target value.

        Arags:
            column: str type. Which column in members dataset to be plotted.

        Return:

        """

        assert column in self.members_raw.columns

        members_train = pd.merge(left=self.members_raw, right=self.train_raw, how='inner', on='msno')
        plt.figure(figsize=(16, 9))
        sns.kdeplot(members_train.loc[members_train.target == 0, column], shade=True, label='0')        
        sns.kdeplot(members_train.loc[members_train.target == 1, column], shade=True, label='1')
        plt.title('{} Distribution by target'.format(column))
        plt.legend(loc='upper right')
        plt.savefig('./figures/kdeplot_members_column_by_target-{}'.format(column))

    def barplot_songs_column(self, column, horizontal=True):
        """Draw barplot.
        
        Draw barplot about columns in songs dataset. Visualize distributions of column data.

        Arags:
            column: str type. Which column in songs dataset to be plotted.
            horizontal: bool type. wheter the plot is horizontal or not.

        Return:

        """

        assert column in self.songs_raw.columns

        plt.figure(figsize=(16, 9))
        if horizontal:
            sns.countplot(y=column, data=self.songs_raw, order=self.songs_raw[column].value_counts().index)
        else:
            sns.countplot(x=column, data=self.songs_raw, order=self.songs_raw[column].value_counts().index)
        plt.title('{} Distribution'.format(column))
        plt.savefig('./figures/barplot_songs_column-{}'.format(column))

    def barplot_songs_column_by_target(self, column, horizontal=True):
        """Draw barplot.
        
        Draw barplot about columns in songs dataset. Visualize distributions of column data based on traget value.

        Arags:
            column: str type. Which column in songs dataset to be plotted.
            horizontal: bool type. wheter the plot is horizontal or not.

        Return:

        """

        assert column in self.songs_raw.columns

        songs_train = pd.merge(left=self.songs_raw, right=self.train_raw, how='inner', on='song_id')
        plt.figure(figsize=(16, 9))
        if horizontal:
            sns.countplot(y=column, hue='target', data=songs_train, order=self.songs_raw[column].value_counts().index)
        else:
            sns.countplot(x=column, hue='target', data=songs_train, order=self.songs_raw[column].value_counts().index)
        plt.title('{} Distribution by target'.format(column))
        plt.legend(loc='upper right')
        plt.savefig('./figures/barplot_songs_column_by_target-{}'.format(column))

    def kdeplot_songs_column(self, column):
        """Draw kdeplot.
        
        Draw kdeplot about columns in songs dataset. Visualize distributions of column data.

        Arags:
            column: str type. Which column in songs dataset to be plotted.

        Return:

        """

        assert column in self.songs_raw.columns

        plt.figure(figsize=(16, 9))
        sns.kdeplot(self.songs_raw[column], shade=True)
        plt.title('{} Distribution by target'.format(column))
        plt.savefig('./figures/kdeplot_songs_column-{}'.format(column))

    def kdeplot_songs_column_by_target(self, column):
        """Draw kdeplot.
        
        Draw kdeplot about columns in songs dataset. Visualize distributions of column data based on target value.

        Arags:
            column: str type. Which column in songs dataset to be plotted.

        Return:

        """

        assert column in self.songs_raw.columns

        songs_train = pd.merge(left=self.songs_raw, right=self.train_raw, how='inner', on='song_id')
        plt.figure(figsize=(16, 9))
        sns.kdeplot(songs_train.loc[songs_train.target == 0, column], shade=True, label='0')        
        sns.kdeplot(songs_train.loc[songs_train.target == 1, column], shade=True, label='1')
        plt.title('{} Distribution by target'.format(column))
        plt.legend(loc='upper right')
        plt.savefig('./figures/kdeplot_songs_column_by_target-{}'.format(column))


if __name__ == '__main__':
    worker = EDAWorker(data_dirname='./data', font_path='./static/fonts/D2Coding.ttc')
    print('Train Dataset Shape: {}'.format(worker.train_raw.shape))
    print('Test Dataset Shape: {}'.format(worker.test_raw.shape))

    print('\n*********Train Dataset Missing Values*********')
    print(worker.train_raw.isna().sum())
    print()

    worker.barplot_train_column_by_target('source_system_tab')
    worker.barplot_train_column_by_target('source_screen_name')
    worker.barplot_train_column_by_target('source_type')

    print('\n*********Members Dataset Missing Values*********')
    print(worker.members_raw.isna().sum())
    print()

    worker.barplot_members_column_by_target('city')
    worker.barplot_members_column_by_target('gender', horizontal=False)
    worker.barplot_members_column_by_target('registered_via', horizontal=False)

    worker.barplot_members_column('registered_via', horizontal=False)

    worker.kdeplot_members_column('bd')
    worker.kdeplot_members_column_by_target('bd')

    print('\n*********Songs Dataset Missing Values*********')
    print(worker.songs_raw.isna().sum())
    print()

    worker.barplot_songs_column_by_target('language')

    worker.barplot_songs_column('language')

    worker.kdeplot_songs_column('song_length')
    worker.kdeplot_songs_column_by_target('song_length')
