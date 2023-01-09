# LEVEL3 U Stage - Movie Recommendation
![image](https://user-images.githubusercontent.com/62127798/211243332-7b1f69c1-fddd-44f8-9a77-04fe8be7be1a.png)

> **목차**
> 
> [팀원 소개](https://github.com/boostcampaitech4lv23recsys1/level2_movierecommendation_recsys-level3-recsys-02/edit/main/README.md#%EF%B8%8F%EF%B8%8F--%ED%8C%80%EC%9B%90-%EC%86%8C%EA%B0%9C)
> 
> [최종 결과](https://github.com/boostcampaitech4lv23recsys1/level2_movierecommendation_recsys-level3-recsys-02/edit/main/README.md#%EC%B5%9C%EC%A2%85-%EA%B2%B0%EA%B3%BC) 
> 
> [프로젝트 개요](https://github.com/boostcampaitech4lv23recsys1/level2_movierecommendation_recsys-level3-recsys-02/edit/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94) 
> 
> [프로젝트 구조](https://github.com/boostcampaitech4lv23recsys1/level2_movierecommendation_recsys-level3-recsys-02/edit/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B5%AC%EC%A1%B0) 
> 
> [설치 및 시작](https://github.com/boostcampaitech4lv23recsys1/level2_movierecommendation_recsys-level3-recsys-02/edit/main/README.md#%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%8B%9C%EC%9E%91) 
>

## 2️⃣ RecSys_2조 2️⃣

### 팀원 소개
<table align="center">
  <tr height="155px">
    <td align="center" width="150px">
      <a href="https://github.com/ktasha45"><img src="https://avatars.githubusercontent.com/ktasha45"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/NIckmin96"><img src="https://avatars.githubusercontent.com/NIckmin96"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/parkkyungjun"><img src="https://avatars.githubusercontent.com/parkkyungjun"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/HeeJeongOh"><img src="https://avatars.githubusercontent.com/HeeJeongOh"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/yhw991228"><img src="https://avatars.githubusercontent.com/yhw991228"/></a>
    </td>
  </tr>
  <tr height="80px">
    <td align="center" width="150px">
      <a href="https://github.com/ktasha45">김동영_4028</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/NIckmin96">민복기_T4074</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/parkkyungjun">박경준_T4076</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/HeeJeongOh">오희정_T4129</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/yhw991228">용희원_T4130</a>
    </td>
  </tr>
</table>
&nbsp;

### 최종 결과
- public
<img width="1085" alt="스크린샷 2023-01-07 오후 2 34 32" src="https://user-images.githubusercontent.com/79159191/211132944-706c1cc7-3409-43d7-9664-85af05d06c58.png">

- private
<img width="1085" alt="스크린샷 2023-01-07 오후 2 34 54" src="https://user-images.githubusercontent.com/79159191/211132945-aac53710-1835-4620-86da-4f5ece75d6d8.png">


### 프로젝트 개요

- **프로젝트 주제**
    
    사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 것
    
    - timestamp를 고려한 사용자의 순차적인 이력
    - implicit feedback
    - 아이템 (영화)과 관련된 content (side-information)의 활용방안
    
- **프로젝트 개요**
    - **input:** user의 implicit 데이터, item(movie)의 meta데이터
    - **output:** user에게 추천하는 item을 user, item이 ','로 구분된 파일(csv) 로 제출
    - **metrics** : normalized Recall@K
- **데이터셋**
    - 원본 데이터가 있다면 특정 시점 이후의 데이터 (sequential)와 특정 시점 이전의 일부 데이터(static) 데이터를 임의로 추출하여, 정답 (ground-truth) 데이터로 사용
    - 데이터셋 구조
        
        ```python
        train
        ├── Ml_item2attributes.json      # item과 genre의 mapping 데이터
        ├── directors.tsv                # item, director
        ├── genres.tsv                   # item, genre -> 1:N
        ├── titles.tsv                   # item, title
        ├── train_ratings.csv            # user, item, timestamp
        ├── writers.tsv                  # item, writer
        └── years.tsv                    # item, year
        ```
        

### 프로젝트 구조

- 베이스라인 구조 (대회 제공 )
    
    ![image](https://user-images.githubusercontent.com/62127798/211243264-dc7b06c9-4527-4c6a-a171-a7ebbc3adb7f.png)
    
    ```python
    code
    ├── datasets.py                        #
    │    ├── PretrainDataset
    │    └── SASRecDataset
    │
    ├── models.py                          #
    │    └── S3RecModel
    ├── modules.py                         #
    │    ├── LayerNorm
    │    ├── Embeddings
    │    ├── SelfAttention
    │    ├── Intermediate
    │    ├── Layer
    │    └── Encoder
    │
    ├── trainers.py                        #
    │    ├── Trainer
    │    ├── PretrainTrainer
    │    └── FinetuneTrainer
    │
    ├── inference.py                       #
    │
    ├── preprocessing.py                   #
    ├── utils.py                           #
    │
    ├── run_pretrain.py                    #
    ├── run_train.py                       #
    ├── sample_submission.ipynb            #
    ├── requirements.txt                   #
    └── output
      └── most_popular_submission.csv
    
    ```
    
- RecBole 베이스라인 구조
    
    ```python
    
    ```
    

### 설치 및 시작

- 베이스라인
    
    ```bash
    ### Installation
    conda create -n movie_rec
    conda activate movie_rec
    pip install -r requirements.txt
    
    ### Pretraining
    python run_pretrain.py
    
    ### Fine-tuning (Main Training)
    # without pre-trained weight
    python run_train.py
    # with pre-trained weight
    python run_train.py --using_pretrain
    
    ### Inference
    python inference.py
    ```
    
- RecBole
    
    ```bash
    
    ```

    
