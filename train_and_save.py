import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from chatglm.configuration_chatglm import ChatGLMConfig
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer

global tokenizer, config, model


class TrainDataset(Dataset):

    def __init__(self, train_data, max_seq_length=512):
        super().__init__()
        self.data = train_data
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        context, target = self.data[index]
        context_ids = tokenizer.encode(context, max_length=self.max_seq_length, truncation=True)
        target_ids = tokenizer.encode(target, max_length=self.max_seq_length, truncation=True, add_special_tokens=False)
        input_ids = context_ids + target_ids + [tokenizer.eos_token_id]
        return {'input_ids': input_ids, 'context_len': len(context_ids)}

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    len_ids = [len(d["input_ids"]) for d in batch]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, d in sorted(zip(len_ids, batch), key=lambda x: -x[0]):
        ids = d["input_ids"]
        context_len = d["context_len"]

        labels = (
                [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (longest - length)
        )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

        ids = ids + [tokenizer.pad_token_id] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


if __name__ == '__main__':
    # 初始化模型
    config = ChatGLMConfig.from_pretrained("chatglm/config.json")
    model = ChatGLMForConditionalGeneration(config=config, empty_init=False).bfloat16()
    tokenizer = ChatGLMTokenizer.from_pretrained("chatglm")

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # 准备训练数据
    train_data = [
        ('你好', 'hello'),
        ('苹果', 'apple')
    ]
    train_dataset = TrainDataset(train_data=train_data)
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=2)

    # 开始训练
    LR = 1e-2
    NUM_EPOCHS = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.detach().float()
            print(f"epoch={epoch}, step={step}, loss={loss}")
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 保存权重
    torch.save(model.state_dict(), "model/model.weights")
