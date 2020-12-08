from bert_model.utils import load_data
import kashgari
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

train_x, train_y = load_data('train')
valid_x, valid_y = load_data('validate')
test_x, test_y = load_data('test')
# print(train_x)
# print(f"train data count: {len(train_x)}")
# print(f"validate data count: {len(valid_x)}")
# print(f"test data count: {len(test_x)}")
model_folder = '/Users/mesie/python/nlp/chinese_L-12_H-768_A-12'
bert_embed = BertEmbedding(model_folder)
model = BiLSTM_CRF_Model(bert_embed)
model.fit(train_x,
          train_y,
          x_validate=valid_x,
          y_validate=valid_y,
          epochs=20,
          batch_size=512)