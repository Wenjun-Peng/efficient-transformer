import os


def load_file(path, label_map):
    all_text = []
    all_label = []
    with open(path, 'r') as f:
        for line in f:
            news_info = line.strip().split('\t')
            label_name = news_info[1]
            if label_name not in label_map:
                label_map[label_name] = len(label_map)
            text = news_info[2]
            all_text.append(text)
            all_label.append(label_map[label_name])
    return all_text, all_label, label_map


def load_mind(root):
    train_path = os.path.join(root, 'train/news.tsv')
    test_path = os.path.join(root, 'test/news.tsv')
    dev_path = os.path.join(root, 'dev/news.tsv')

    label_map = {}

    dataset = {'train': {'text': [], 'label': []},
               'test': {'text': [], 'label': []},
               'dev': {'text': [], 'label': []}}

    all_text, all_label, label_map = load_file(train_path, label_map)
    dataset['train']['text'] = all_text
    dataset['train']['label'] = all_label

    all_text, all_label, label_map = load_file(test_path, label_map)
    dataset['test']['text'] = all_text
    dataset['test']['label'] = all_label

    all_text, all_label, label_map = load_file(dev_path, label_map)
    dataset['dev']['text'] = all_text
    dataset['dev']['label'] = all_label

    return dataset