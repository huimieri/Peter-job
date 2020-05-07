from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl


def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

# 因为是词向量训练，所以这里我们这里把包括y在内的所有的数据以及测试集数据都读进来
def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=100):
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # 训练模型，LineSentence是读取句子的方法
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=5)
    # 保存模型
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # 词向量模型效果测试
    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)
    # 加载我们的词向量模型，这个模型之前是以二进制形式保存的，加载的包再KeyedVectors里
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('./train_set.seg_x.txt',
          './train_set.seg_y.txt',
          './test_set.seg_x.txt',
          out_path='./word2vec.txt',
          sentence_path='./sentences.txt',)