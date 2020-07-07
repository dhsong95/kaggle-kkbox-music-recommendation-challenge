"""Preprocessing WSDM Dataset.

Author: DHSong
Last Modified At: 2020.07.06

Preprocessing WSDM Dataset.
"""

import os
from collections import Counter
from tqdm import tqdm
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

class PreprocessingWorker:
    """Worker for Preprocessing.

    Worker for Preprocessing.

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
    
    def _barplot(self, df, column, horizontal=True):
        plt.figure(figsize=(16, 9))
        if horizontal:
            sns.countplot(y=column, hue='target', data=df, order=df[column].value_counts().index)
        else:
            sns.countplot(x=column, hue='target', data=df, order=df[column].value_counts().index)
        plt.title('{} Distribution by target'.format(column))
        plt.legend(loc='upper right')
        plt.savefig('./figures/preprocessing-barplot-{}'.format(column))

    def preprocess_train_test(self):
        """Preprocess train.csv and test.csv.
        
        preprocess train.csv and test.csv. Select values to be considered.

        Arags:

        Return:
            train: Pandas Dataframe. Select values to be considered in train.csv.
            test: Pandas Dataframe. Select values to be considered in test.csv.
        """

        train = self.train_raw.fillna('<blank>')
        test = self.test_raw.fillna('<blank>')

        selected_values_by_columns = {
            'source_system_tab': [
                '<blank>', '<not selected>', 
                'my library', 'discover', 'search', 'radio'
            ], 
            'source_screen_name': [
                '<blank>', '<not selected>',
                'Local playlist more', 'Online playlist more', 'Radio', 
                'Album more', 'Search', 'Artist more', 'Discover Feature', 
                'Discover Chart', 'Others profile more'
            ], 
            'source_type': [
                '<blank>', '<not selected>',
                'local-library', 'online-playlist', 'local-playlist', 
                'radio', 'album', 'top-hits-for-artist'
            ]
        }

        for column, values in selected_values_by_columns.items():
            train.loc[~train[column].isin(values), column] = '<not selected>'
            test.loc[~test[column].isin(values), column] = '<not selected>'


        for column in selected_values_by_columns.keys():
            self._barplot(train, column)

        return train, test

    def preprocess_members(self):
        """Preprocess members.csv.
        
        preprocess members.csv. Select values to be considered.

        Arags:

        Return:
            members: Pandas Dataframe. Select values to be considered in members.csv.
        """

        # fill all the NA with <blank>.
        members = self.members_raw.fillna('<blank>')

        # calculate membership days.
        members['registration_init_time'] = pd.to_datetime(members.registration_init_time, format='%Y%m%d')
        members['expiration_date'] = pd.to_datetime(members.expiration_date, format='%Y%m%d')
        members['membership_days'] = (members.expiration_date - members.registration_init_time).dt.days
        
        # binning membership days.
        invalid_membership_days = members.membership_days < 0
        members.loc[invalid_membership_days, 'membership_days'] = -1
        members.loc[invalid_membership_days, 'membership_dyas_bin'] = '<invalid>'
        members.loc[~invalid_membership_days, 'membership_dyas_bin'] = pd.qcut(members.loc[~invalid_membership_days, 'membership_days'], 3)

        # binning bd(age).
        invalid_bd = (members.bd < 0) | (members.bd >= 100)
        members.loc[invalid_bd, 'bd'] = -1
        members.loc[invalid_bd, 'bd_bin'] = '<invalid>'
        members.loc[~invalid_bd, 'bd_bin'] = pd.cut(members.loc[~invalid_bd, 'bd'], 5)

        selected_values_by_columns = {
            'city': [
                '<blank>', '<not selected>',
                '1', '13', '5', '4', '15', '22'
            ],
            'registered_via': [
                '<blank>', '<not selected>', 
                '4', '7', '9', '3'
            ]
        }

        for column, values in selected_values_by_columns.items():
            members[column] = members[column].astype('str')
            members.loc[~members[column].isin(values), column] = '<not selected>'

        members_train = pd.merge(left=members, right=self.train_raw, how='inner')
        for column in selected_values_by_columns.keys():
            self._barplot(members_train, column)
        
        self._barplot(members_train, 'bd_bin')
        self._barplot(members_train, 'gender')
        self._barplot(members_train, 'membership_dyas_bin')

        return members

    def preprocess_songs(self):
        """Preprocess songs.csv.
        
        preprocess songs.csv. Select values to be considered.

        Arags:

        Return:
            songs: Pandas Dataframe. Select values to be considered in songs.csv.
        """

        # fill all the NA with <blank>.
        songs = self.songs_raw.fillna('<blank>')

        # binning song length.
        invalid_song_length = songs.song_length < 0
        songs.loc[invalid_song_length, 'song_length'] = -1
        songs.loc[invalid_song_length, 'song_length_bin'] = '<invalid>'
        songs.loc[~invalid_song_length, 'song_length_bin'] = pd.qcut(songs.loc[~invalid_song_length, 'song_length'], 3)
        
        # select only top genres.
        genre_list = list()
        for genres in tqdm(songs.genre_ids.str.split('|')):
            for genre in genres:
                if genre.isdigit():
                    genre_list.append(genre)
        counter = Counter(genre_list)
        top_genres = [genre for genre, freq in counter.most_common(5)]

        for genre in tqdm(top_genres):
            name = 'genre_{}'.format(genre)
            values = list()

            for genres in songs.genre_ids:
                value = 0
                if genre in genres.split('|'):
                    value = 1
                values.append(value)

            songs[name] = values

        selected_values_by_columns = {
            'language': [
                '<blank>', '<not selected>',
                '52', '3', '27', '24', '31', '10'
            ]
        }

        songs.loc[songs.language == '<blank>', 'language'] = -1
        songs['language'] = songs.language.astype('int')
        songs.loc[songs.language == -1, 'language'] = '<blank>'

        for column, values in selected_values_by_columns.items():
            songs[column] = songs[column].astype('str')
            songs.loc[~songs[column].isin(values), column] = '<not selected>'

        songs_train = pd.merge(left=songs, right=self.train_raw, how='inner')
        for column in selected_values_by_columns.keys():
            self._barplot(songs_train, column)
        
        self._barplot(songs_train, 'song_length_bin')
        for genre in top_genres:
            name = 'genre_{}'.format(genre)
            self._barplot(songs_train, name)

        return songs

    def preprocess_song_extra_info(self):
        """Preprocess songs.csv.
        
        preprocess songs.csv. Select values to be considered.

        Arags:

        Return:
            songs: Pandas Dataframe. Select values to be considered in songs.csv.
        """

        def isrc_to_country(isrc):
            if isrc != '<blank>':
                return isrc[:2]
            else:
                return isrc

        def isrc_to_year(isrc):
            if isrc != '<blank>':
                year = isrc[5:7]
                if int(year) > 18:
                    return int('19' + year)
                else:
                    return int('20' + year)
            else:
                return -1

        # fill all the NA with <blank>.
        song_extra_info = self.song_extra_info_raw.fillna('<blank>')

        song_extra_info['country'] = song_extra_info.isrc.apply(lambda x: isrc_to_country(x))
        song_extra_info['year'] = song_extra_info.isrc.apply(lambda x: isrc_to_year(x))

        blank_year = (song_extra_info.year == -1)
        song_extra_info.loc[blank_year, 'year_bin'] = '<blank>'
        song_extra_info.loc[~blank_year, 'year_bin'] = pd.qcut(song_extra_info.loc[~blank_year, 'year'], 5)

        selected_values_by_columns = {
            'country': [
                '<blank>', 
                'US', 'GB', 'DE', 
                'FR', 'TC', 'JP'
            ]
        }

        for column, values in selected_values_by_columns.items():
            song_extra_info[column] = song_extra_info[column].astype('str')
            song_extra_info.loc[~song_extra_info[column].isin(values), column] = '<not selected>'

        song_extra_info_train = pd.merge(left=song_extra_info, right=self.train_raw, how='inner')
        for column in selected_values_by_columns.keys():
            self._barplot(song_extra_info_train, column)
        
        self._barplot(song_extra_info_train, 'year_bin')

        return song_extra_info


if __name__ == '__main__':
    worker = PreprocessingWorker(data_dirname='./data', font_path='./static/fonts/D2Coding.ttc')
    print('*****Preprocess train.csv and test.csv*****')
    train, test = worker.preprocess_train_test()
    print('\n*****Preprocess members.csv*****')
    members = worker.preprocess_members()
    print('\n*****Preprocess songs.csv*****')
    songs = worker.preprocess_songs()
    print('\n*****Preprocess song_extra_info.csv*****')
    song_extra_info = worker.preprocess_song_extra_info()

    members = members.drop(columns=['bd', 'registration_init_time', 'expiration_date', 'membership_days'])
    songs = songs.drop(columns=['song_length', 'genre_ids', 'artist_name', 'composer', 'lyricist'])
    song_extra_info = song_extra_info.drop(columns=['name', 'isrc', 'year'])

    merged = pd.merge(left=train, right=members, how='left', on='msno')
    merged = pd.merge(left=merged, right=songs, how='left', on='song_id')
    merged = pd.merge(left=merged, right=song_extra_info, how='left', on='song_id')
    merged.to_csv('./data/train_merged.csv', index=False)

    merged = pd.merge(left=test, right=members, how='left', on='msno')
    merged = pd.merge(left=merged, right=songs, how='left', on='song_id')
    merged = pd.merge(left=merged, right=song_extra_info, how='left', on='song_id')
    merged.to_csv('./data/test_merged.csv', index=False)