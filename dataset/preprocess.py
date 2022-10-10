import os
import pandas as pd
from transformers import BertTokenizer


def load_transcript(libri_root, x_list):
    """Load the transcripts for Librispeech"""

    def process_trans(transcript):
        # TODO: support character / bpe
        transcript = transcript.upper()
        return transcript

    trsp_sequences = {}
    split_spkr_chap_list = list(
        set(
            "/".join(x.split('/')[:-1]) for x in x_list
        )
    )

    for dir in split_spkr_chap_list:
        parts = dir.split('/')
        trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
        path = os.path.join(libri_root, dir, trans_path)
        assert os.path.exists(path)

        with open(path, "r") as trans_f:
            for line in trans_f:
                lst = line.strip().split()
                trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))

    return trsp_sequences


if __name__ == "__main__":
    args = {
        'test-clean': ['test-clean'],
        'bucket_size': 10,
        'bucket_file': './dataset/data'
    }
    libri_root='F:\OneDrive\数据集\Librispeech\\test-clean\LibriSpeech'
    bucket_dir='./data'
    bucket_file = ['test-clean']
    table_list = []
    for file in bucket_file:
        file_path = os.path.join(bucket_dir, (file + '.csv'))
        if os.path.exists(file_path):
            table_list.append(
                pd.read_csv(file_path)
            )

    print('get {} files'.format(len(table_list)))

    table_list = pd.concat(table_list)
    table_list = table_list.sort_values(by=['length'], ascending=False)

    X = table_list['file_path'].tolist()
    X_lens = table_list['length'].tolist()

    # Transcripts
    Y = load_transcript(libri_root, X)
    t = Y['672-122797-0033']
    tokenizer = BertTokenizer.from_pretrained('../pretrain_models/bert-base-cased')
    text_token = tokenizer(t, return_tensors='pt')
    print('done')
