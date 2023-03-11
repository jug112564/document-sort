import json
import pandas as pd
import os


def make_csv(my_path, my_file_name):
    # column 이름 미리 지정
    df = pd.DataFrame(columns=['passage', 'doc_type'])
    # json 파일들 경로 저장
    dir_path = my_path
    paths = []

    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            paths.append(file_path)

    # json 파일 1개씩 읽어오기
    for i in paths:
        with open(i, "r", encoding="utf8") as input_file:
            json_data = json.load(input_file)
            data = {'passage': [json_data['Meta(Refine)']['passage']],
                    'doc_type': [json_data['Meta(Acqusition)']['doc_type']]}
            data = pd.DataFrame(data)
            df = pd.concat([df, data])

    # csv로 변환하기
    df.to_csv(my_file_name)


make_csv('c:/train_json/', 'c:/csv/train_data.csv')
make_csv('c:/test_json/', 'c:/csv/test_data.csv')

# print(json.dumps(json_data, indent='\t'))