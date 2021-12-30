from dataset import KoBARTSummaryDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import os
import argparse
from scorer import RougeScorer

def load_model(checkpoint_path):
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    # tokenizer = get_kobart_tokenizer()
    return model


def test1(checkpoint_path, batch_size=8, num_workers=1, max_len=512, beam_size=1):
    tok = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

    scorer = RougeScorer()

    model = load_model(checkpoint_path)
    model.cuda()

    # train_file_path = "./data/train.tsv"
    test_file_path = "./data/test.tsv"

    # train = KoBARTSummaryDataset(train_file_path, tok, max_len)
    test = KoBARTSummaryDataset(test_file_path, tok, max_len)

    test_data_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    out_filename = os.path.join(checkpoint_path,"beam_size%d_output.txt" % beam_size)
    ref_filename = os.path.join(checkpoint_path,"beam_size%d_reference.txt" % beam_size)

    f1 = open(out_filename, "w")
    f2 = open(ref_filename, "w")

    model.eval()
    for batch in test_data_loader:
        input_ids = batch['input_ids']

        labels = batch['labels']

        input_ids = input_ids.cuda()

        outputs = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=beam_size)

        outputs = outputs.data.cpu().numpy().tolist()

        labels = labels.data.cpu().numpy().tolist()

        for output, label in zip(outputs, labels):
            output = tok.decode(output, skip_special_tokens=True)

            label = [3 if item == -100 else item for item in label ]

            label = tok.decode(label, skip_special_tokens=True)

            output = output.replace("\n", "")
            label = label.replace("\n", "")

            f1.write(output + "\n")
            f2.write(label + "\n")

    f1.close()
    f2.close()

    def read_data(filename):
        datas = []

        for line in open(filename):
            line = line.strip()
            data = " ".join(tok.tokenize(line))

            datas.append(data)

        return datas

    reference_summaries = read_data(ref_filename)
    generated_summaries = read_data(out_filename)

    scores = scorer.rouge_evaluator.get_scores(generated_summaries, reference_summaries)

    print(scorer.format_rouge_scores(scores))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path', required=True)
    parser.add_argument('--beam_size', type=int, default=1, help='')

    args = parser.parse_args()

    test1(args.checkpoint_path, batch_size=8, num_workers=1, max_len=512, beam_size=args.beam_size)