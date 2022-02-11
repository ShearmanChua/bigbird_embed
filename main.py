from transformers import BigBirdModel
from transformers import AutoTokenizer
import torch
torch.cuda.is_available()
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tempfile import gettempdir
from clearml import Task, Logger
from tqdm import tqdm

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    task = Task.init(project_name='bigbird', task_name='bigbird_embedding_task')
    logger = task.get_logger()
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    # task.execute_remotely(queue_name="compute", exit_process=True)
    df = pd.read_csv('300_texts_cleaned.csv')
    docs = df['cleaned_texts'].tolist()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-large')
    model = BigBirdModel.from_pretrained("google/bigbird-roberta-large")

    class TrainDataset(Dataset):
        def __init__(self, txt):
            self.texts = txt
        def __len__(self):
                return len(self.texts)
        def __getitem__(self, idx):
                text = self.texts[idx]
                sample = {"Text": text}
                return sample

    train_dataset = TrainDataset(docs)
    train_loader = DataLoader(train_dataset, batch_size=1)
    embeddings = []

    pbar = tqdm(train_dataset.__len__, position=0, leave=True)
    doc_count = 0
    for batch in train_loader:
    
        docs = batch['Text']

        encoded_input = tokenizer(docs, padding=True,truncation=True,return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.append(sentence_embeddings.cpu().detach().numpy()[0])
        pbar.update(1)
        doc_count += 1
        pbar.set_description('Documents processed: {}/{}'.format(doc_count,train_dataset.__len__))
        logger.report_text('Documents processed: {}/{}'.format(doc_count,train_dataset.__len__))


    df['embeddings'] = embeddings
    df.to_csv("300_texts_embbed.csv",index=False)



if __name__ == '__main__':
    main()
