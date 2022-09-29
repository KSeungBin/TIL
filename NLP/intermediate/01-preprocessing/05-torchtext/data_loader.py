from torchtext import data


class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(
        self, train_fn, # train_filename
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=False, # <end of sentence> token 사용하지 않는다
        shuffle=True   # train : shuffle = True / valid & test : shuffle = False
    ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()

        # Define field of the input file.
        # The input file consists of two fields.
        self.label = data.Field(
            sequential=False,  # class만 있기 떄문에 sequential data 아니다
            use_vocab=True,    # class(positive/negative)를 하나의 단어로 취급
            unk_token=None
        )
        self.text = data.Field(
            use_vocab=True,
            batch_first=True,      # batch dimension을 맨 앞에 위치시키는 것 추천
            include_lengths=False, # NLG에서는 include_lengths, eos_token -> True
            eos_token='<EOS>' if use_eos else None
        )

        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # We would have two separate input file: train_fn, valid_fn
        # Files consist of two columns: label field and text field.
        train, valid = data.TabularDataset(
            path=train_fn,
            format='tsv', 
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],
        ).split(split_ratio=(1 - valid_ratio))

        # Those loaded dataset would be feeded into each iterator:
        # train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.
        self.train_loader, self.valid_loader = data.BucketIterator.splits(  # dataset 구축되면 DataLoader에 넣는다
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            sort_key=lambda x: len(x.text), # 비슷한 길이의 문장끼리 mini-batch로 구성하기 위해 text의 길이를 기준으로 정렬                    
            sort_within_batch=True,         # NLG에서는 반드시 True : mini-batch 내부에서도 text의 길이를 기준으로 정렬된다.
        )

        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)  # vocab 2개 : positive/negative
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)