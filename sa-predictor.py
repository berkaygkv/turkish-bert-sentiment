import torch
import torch.nn as nn
from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np


class SentimentPredictor:
    def __init__(self) -> None:
        self.label_encoder_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        self.device = torch.device("mps")
        self.tokenizer = BertTokenizer.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased", do_lower_case=False
        )
        self.custom_bert = torch.load("model/Turkish-SA-model.pth")

    def read_text_file(self):
        with open("input_text.txt", "r") as rd:
            data = rd.read()
            return data

    def prediction_pipeline(self, texts):
        input_ids = []
        attention_masks = []
        for text in texts:
            text = text.replace("\n", " ").strip()
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=500,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        with torch.no_grad():
            device_input_ids = input_ids.to(self.device)
            device_attention_masks = attention_masks.to(self.device)
            outputs = self.custom_bert(
                device_input_ids,
                token_type_ids=None,
                attention_mask=device_attention_masks,
            )

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        probs = probs.detach().cpu().numpy()
        label = np.argmax(probs, axis=1).flatten()

        vfunc = np.vectorize(lambda x: self.label_encoder_mapping[x])
        label = vfunc(label)

        prob = np.max(probs, axis=1).flatten()
        prob = np.round(prob, 3)
        if len(label) == 1:
            print(f"Text:\n{text}\n\nLabel: {label[0]}\nProb: {prob[0]:.2f}")
        return label, prob


if __name__ == "__main__":
    predictor = SentimentPredictor()
    text = predictor.read_text_file()

    import sqlite3
    import pandas as pd

    conn = sqlite3.connect("data/crawled_data.db")
    df = pd.read_sql("SELECT * FROM enuygun_googleplay LIMIT 150", conn)

    # label, prob = predictor.prediction_pipeline([text])
    labels, probs = predictor.prediction_pipeline(df["text"].tolist())
    df = df.assign(prediction=labels).assign(prob=probs)
    df[["text", "prediction", "prob"]].to_excel("output/googleplay_predictions.xlsx", index=False)
    print(df[["text", "prediction", "prob"]].head(50))
