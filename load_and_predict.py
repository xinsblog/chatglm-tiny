import torch

from chatglm.configuration_chatglm import ChatGLMConfig
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer


if __name__ == '__main__':
    # 初始化模型
    config = ChatGLMConfig.from_pretrained("chatglm/config.json")
    model = ChatGLMForConditionalGeneration(config=config, empty_init=False).bfloat16()
    tokenizer = ChatGLMTokenizer.from_pretrained("chatglm")

    # 加载权重
    model.load_state_dict(torch.load("model/model.weights"))

# 准备测试数据
test_data = [
    ('你好', 'hello'),
    ('苹果', 'apple')
]

# 测试生成效果
model = model.eval()
for query, _ in test_data:
    response, history = model.chat(tokenizer, query, history=[], max_length=20)
    print(query, response)





