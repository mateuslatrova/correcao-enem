import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from transformers import AutoModel


class TopicDeviationNeuralNetwork(nn.Module):
    def __init__(self, checkpoint) -> None:
        super(TopicDeviationNeuralNetwork, self).__init__()
        self.model = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Sequential(nn.Linear(1536, 2), nn.Sigmoid())

    def forward(self, essay_text, topic_text, labels=None):
        outputs = self.model(**essay_text)
        last_hidden_states = outputs.last_hidden_state
        essay_representations = last_hidden_states[:, 0, :]

        outputs = self.model(**topic_text)
        last_hidden_states = outputs.last_hidden_state
        topic_representations = last_hidden_states[:, 0, :]

        joint_representations = torch.cat((essay_representations, topic_representations), dim=1)

        logits = self.classifier(joint_representations)

        training = labels is not None

        if training:
            loss = cross_entropy(logits, labels["labels"])
            outputs = {"logits": logits, "loss": loss}
        else:
            outputs = {"logits": logits, "loss": None}

        return outputs
