Exploratory Data Analysis
==============================

## 1. Data Description

Kaggle Competition 중 하나인 [KKBOX Music Recommendataion Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge) 문제를 해결한다. Kaggle에서 제공하는 데이터 파일은 총 6개이다. 데이터 파일은 csv 파일이 압축된 형태로 제공된다. 

|파일 이름|열(column) 이름|내용|
|-------|-------------|---|
|trian.csv|msno|사용자 ID|
||song_id|노래 ID|
||source_system_tab|이벤트가 발생한 탭으로 KKBox 모바일 애플리케이션의 기능|
||source_screen_name|사용자가 사용한 화면 레이아웃|
||source_type|모바일 애플리케이션에서 사용자가 노래를 재생한 지점|
||target|한 달 내에 사용자가 노래를 다시 듣는지 여부|
|test.csv|id|레코드 ID|
||msno|사용자 ID|
||song_id|노래 ID|
||source_system_tab|이벤트가 발생한 탭으로 KKBox 모바일 애플리케이션의 기능|
||source_screen_name|사용자가 사용한 화면 레이아웃|
||source_type|모바일 애플리케이션에서 사용자가 노래를 재생한 지점|
|sample_submission.csv|id|test.csv의 레코드 ID|
||target|한 달 내에 사용자가 노래를 다시 듣는지 여부. 모델을 통해 이를 예측|
|songs.csv|song_id|노래 ID|
||song_length|노래 길이. ms 단위|
||genre_ids|노래의 장르 ID. 여러 개의 장르에 해당하는 경우 \| 로 구분|
||artist_name|아티스트 이름|
||composer|작곡가|
||lyricst|작사가|
||language|노래 언어|
|members.csv|msno|사용자 ID|
||city|사용자 소속 도시|
||bd|사용자 나이. outlier 주의|
||gender|사용자 성별|
||registered_via|가입 경로|
||registration_init_time|가입 일시. %Y%m%d 형태|
||expiration_date|만료 일시. %Y%m%d 형태|
|song_extra_info.csv|song_id|노래 ID|
||song_name|노래 제목|
||isrc|International Standard Recording Code. 노래 식별을 위해 사용. 검증되지 않았으므로 사용에 유의|

<br>데이터 파일(CSV 파일)에 있는 레코드의 개수는 다음과 같다.

|파일 이름|레코드 개수|
|-------|--------:|
|trian.csv|7,377,418|
|test.csv|2,556,790|
|sample_submission.csv|2,556,790|
|songs.csv|2,296,320|
|members.csv|34,403|
|song_extra_info.csv|2,295,971|


## 2. Basic EDA

### 1. train.csv + test.csv

train.csv의 데이터와 test.csv의 데이터는 target을 제외하고 msno, song_id, source_system_tab, source_screen_name, source_typ 변수를 모두 가지고 있다. train.csv와 test.csv를 통합하여서 해당 변수의 전체적인 분포를 살펴본다. 