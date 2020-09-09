from transformers import AutoTokenizer, AutoModelWithLMHead
from argparse import ArgumentParser
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('t5-large')
model = AutoModelWithLMHead.from_pretrained('t5-large-checkpt')

argparser = ArgumentParser()
argparser.add_argument('-s', '--src')
argparser.add_argument('-t', '--trg')
args = argparser.parse_args()

test_corpus = open(args.src, encoding='utf-8')
corrected_corpus = open(args.trg, 'w', encoding='utf-8')

for line in tqdm(test_corpus):
    input_ids = tokenizer.encode(line.strip(), return_tensors='pt')
    output_ids = model.generate(input_ids)
    corrected = tokenizer.decode(output_ids[0])
    corrected_corpus.write(corrected + '\n')
