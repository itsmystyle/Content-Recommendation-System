import torch.nn as nn

from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertContentBased(BertPreTrainedModel):
    def __init__(self, config, num_class_1, num_class_2, compress_dim=512):
        super(BertContentBased, self).__init__(config)

        self.num_class_1 = num_class_1
        self.num_class_2 = num_class_2
        self.compress_dim = compress_dim

        self.bert = BertModel(config)
        self.feature_encoder = nn.Sequential(nn.Linear(768, self.compress_dim), nn.ReLU(True))
        self.class_1_cls = nn.Sequential(
            nn.Linear(self.compress_dim, self.num_class_1), nn.LogSoftmax(dim=1)
        )
        self.class_2_cls = nn.Sequential(
            nn.Linear(self.compress_dim, self.num_class_2), nn.LogSoftmax(dim=1)
        )

    def forward(self, inputs):
        x = self.extract_feature(inputs)

        class_1_logits = self.class_1_cls(x)
        class_2_logits = self.class_2_cls(x)

        return {"features": x, "class_1": class_1_logits, "class_2": class_2_logits}

    def extract_feature(self, inputs):
        x = self.bert(
            inputs["content"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_masks"],
        )
        x = self.feature_encoder(x[1])

        return x
