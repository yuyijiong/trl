import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from transformers import AutoModelForSeq2SeqLM,AutoModelForSequenceClassification,\
    pipeline,AutoTokenizer,TextClassificationPipeline,DataCollatorForSeq2Seq
from functools import partial
import torch

bert_name='IDEA-CCNL/Erlangshen-Roberta-110M-NLI'#'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI'
bert_model=AutoModelForSequenceClassification.from_pretrained(bert_name)
bert_tokenizer=AutoTokenizer.from_pretrained(bert_name)
if bert_tokenizer.eos_token is None:
    bert_tokenizer.eos_token=bert_tokenizer.sep_token
#pipe_output=sentiment_pipe(['天气好[SEP]天气真不错','天气不太行[SEP]天气真不错'])

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training,PeftModelForSeq2SeqLM,TaskType,PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline,TextClassificationPipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from nlp_utils import T5_utils


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        default="/home/datamining/lxl/JD/mt0-xl-sentiment-quadruple-lora-adapter-merged", metadata={"help": "the model name"}
    )
    log_with: Optional[str] = field(default='tensorboard', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    remove_unused_columns=False,
    accelerator_kwargs={'device_placement':True,'project_dir':'./trl_logs','gradient_accumulation_steps':script_args.gradient_accumulation_steps},
    tracker_project_name='./trl_logs')

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
#mini_batch_size是pipeline的batch_size
sent_kwargs = {"return_all_scores": True,
               "function_to_apply": "none",
               "batch_size": config.mini_batch_size}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    #tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds  = load_dataset('json', data_files=dataset_name)['train']

    ds = ds.map(partial(T5_utils.tokenize_func_QA,tokenizer=tokenizer,need_label=False), batched=False,desc='tokenize train dataset')
    #将question重命名为query
    ds=ds.rename_column("question", "query")
    #不需要标签，因为标签是自己生成的
    ds.set_format(type="torch",columns=['input_ids','attention_mask','query'])
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config, dataset_name="/home/datamining/lxl/JD/dataset_for_relabel/JD_prompt2quadruple_format0_all/comment_labeled.json")


#进行pad
def T5_collator(batch:list,tokenizer:AutoTokenizer):
    #batch中的每个dict只保留query
    batch_query=[{key:val for key,val in item.items() if key in ['query']} for item in batch]
    batch_query=dict((key, [d[key] for d in batch_query]) for key in batch_query[0])
    #batch中的每个dict只保留input_ids和attention_mask
    batch=[{key:val for key,val in item.items() if key in ['input_ids','attention_mask']} for item in batch]
    #将batch进行pad
    batch_ids=tokenizer.pad(batch,padding=True,return_tensors="pt")

    #将batch_query和batch_ids合并
    batch_ids.update(batch_query)
    return batch_ids


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
device_map={'shared':0,'encoder':0,'decoder':0,'lm_head':0}
pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, load_in_8bit=True, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
collator=partial(T5_collator,tokenizer=tokenizer)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q','v'],  # handled automatically by peft
    lora_dropout=0.05,
    bias="lora_only",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

#准备8bit模型预处理，会开启梯度检查点
pretrained_model = prepare_model_for_int8_training(pretrained_model)
#转换为peft模型
pretrained_model = get_peft_model(pretrained_model, lora_config)
pretrained_model.print_trainable_parameters()
#增加value head
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(pretrained_model,device=0)
d1_device=dict([(n, p.device) for n, p in model.named_parameters() if p.device.index==1])

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config, model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": 3,
    'max_new_tokens':64,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    'early_stopping':True,
}

#从pipe获取reward
def get_reward_from_pipe_output(output:list)->torch.tensor:
    #选取output中，label为‘ENTAILMENT'的字典
    entail_dict = [o for o in output if o['label'] == 'ENTAILMENT'][0]
    return torch.tensor(entail_dict['score'])

sentiment_pipe=pipeline('text-classification',model=bert_model,device=1,tokenizer=bert_tokenizer,top_k=None,torch_dtype=torch.float16)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader),total=len(dataset)//config.batch_size,desc='PPO'):
    query_tensors = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    generation_kwargs["attention_mask"] = attention_mask

    #generate的时候需要关闭梯度检查点并use_cache，否则会报错
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True

    #对每个样本进行generate，再把结果拼接为list
    response_tensors = ppo_trainer.model.generate(input_ids=query_tensors, **generation_kwargs)
    response_text=tokenizer.batch_decode(response_tensors,skip_special_tokens=True)
    batch["response"] =response_text# [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # 拼接问题和答案，获取答案得分
    texts = [q + bert_tokenizer.sep_token+r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [get_reward_from_pipe_output(output).to(query_tensors.device) for output in pipe_outputs]

    # Run PPO step
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    #query_tensors\response_tensors拆分为batchsize个tensor
    query_tensors = torch.split(query_tensors, 1, dim=0)
    query_tensors = [q.squeeze() for q in query_tensors]
    response_tensors = torch.split(response_tensors, 1, dim=0)
    response_tensors = [r.squeeze() for r in response_tensors]

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    #记录log
    ppo_trainer.log_stats(stats, batch, rewards)

#结束
ppo_trainer.accelerator.end_training()
model.save_pretrained('aug_T5')
tokenizer.save_pretrained('aug_T5')
