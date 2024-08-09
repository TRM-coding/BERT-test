import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from data_processing2 import *

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')

'''文本向量表示'''
# 使用分词器进行编码，并启用截断和填充
encoded_train_text = tokenizer(
    text=train_text,
    add_special_tokens=True,  # 添加特殊标记，如[CLS]和[SEP]
    max_length=45,  # 根据列表中最长序列进行填充
    truncation=True,  # 不截断文本
    padding=True,
    return_tensors='pt'  # 返回PyTorch张量
)
encoded_train_text['input_ids'].requires_grad
encoded_train_text['token_type_ids'].requires_grad
encoded_train_text['attention_mask'].requires_grad
train_label.requires_grad


encoded_test_text = tokenizer(
    text=test_text,
    add_special_tokens=True,  # 添加特殊标记，如[CLS]和[SEP]
    padding='longest',  # 根据列表中最长序列进行填充
    truncation=False,  # 不截断文本
    return_tensors='pt'  # 返回PyTorch张量
)
#print('input_ids: ' + str(encoded_test_text['input_ids'].shape))  # input_ids: torch.Size([2500, 48])
#print('token_type_ids: ' + str(encoded_test_text['token_type_ids'].shape))  # token_type_ids: torch.Size([2500, 48])
#print('attention_mask: ' + str(encoded_test_text['attention_mask'].shape))  # attention_mask: torch.Size([2500, 48])


class Module(torch.nn.Module):  # 确保继承自 torch.nn.Module
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.batchnorm = torch.nn.BatchNorm1d(num_features=768)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 5)

    def forward(self, encoded1, encoded2, encoded3):
        bert = self.bert(encoded1, encoded2, encoded3)
        #print(bert[0][:, 0].shape) # torch.Size([32, 768])
        #print(bert[0][:, 0, :] == bert[0][:, 0]) # True
        #print(bert)
        #print(bert[1])
        batchnorm = self.batchnorm(bert[1])
        #print(batchnorm)
        dropout = self.dropout(batchnorm)
        linear = self.linear(dropout)
        #print(linear)
        return torch.softmax(linear, dim=1)


model = Module()

for name, param in model.named_parameters():
    #print(name)
    if "bert" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = torch.nn.CrossEntropyLoss()




batch_size = 32 #显存太大，这个数可以调大，32,64，128，256，512
from torch.utils.data import DataLoader
import torch.utils.data as Data

class MyDataset(Data.Dataset):
    def __init__(self, data, label):
        self.input_ids = data["input_ids"]
        self.token_type_ids = data["token_type_ids"]
        self.attention_mask = data["attention_mask"]

        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.label[idx]


loader = DataLoader(MyDataset(encoded_train_text, train_label), batch_size=batch_size, shuffle=True)
loader2 = DataLoader(MyDataset(encoded_test_text, test_label), batch_size=batch_size, shuffle=True)


test_loss = 0
test_accuracy = 0

train_losses = []
train_accuracies = []

for epoch in tqdm(range(10)):
    #pbar = tqdm(loader, total=len(loader))

    model.train()
    #for data in pbar:
    time = 0
    for data in loader:
        optimizer.zero_grad()
        #print(data[0])
        #print(data[1])
        #print(data[2])
        #print(data[3].view(-1))
        logits = model(data[0], data[1], data[2])
        #print(logits)  # torch.Size([32, 5])
        loss = criterion(logits, data[3])
        loss.backward()
        optimizer.step()
        acc = (torch.eq(torch.argmax(logits, dim=1), data[3])).type(torch.float32).sum().item() / batch_size
        time += 1
        print(f"train on batch {time+1}: loss: {loss.item()/batch_size}, acc: {acc}\n")
        train_losses.append(loss.item()/batch_size)
        train_accuracies.append(acc)

model.eval()
with torch.no_grad():
    time = 0
    for data in tqdm(loader):
        logits = model(data[0], data[1], data[2])
        loss = criterion(logits.view(-1, logits.size(-1)), data[3].view(-1)) / batch_size
        test_loss += loss.item()
        acc = (torch.eq(torch.argmax(logits, dim=1), data[3])).type(torch.float32).sum().item() / batch_size
        test_accuracy += acc
        print(f"test on batch {time + 1}: loss: {loss.item()}, acc: {acc}\n")
        time += 1

test_loss /= time
test_accuracy /= time
print(f"test on whole data: loss: {test_loss}, acc: {test_accuracy}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()

plt.show()


torch.save(model.state_dict(), f"./saver/toutiao_model{time+1}.pth")

