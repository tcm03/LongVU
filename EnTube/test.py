from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def main():

    preds = []
    truths = []
    with open('EnTube/partial_res.txt', 'r') as f:
        lines = f.readlines()
        pred_lines = [line for line in lines if line.startswith('pred=')]
        print(f'Number of predictions: {len(pred_lines)}')
        for line in pred_lines:
            # format: pred=0, truth=0
            pred = int(line.split(',')[0].split('=')[1].strip())
            truth = int(line.split(',')[1].split('=')[1].strip())
            preds.append(pred)
            truths.append(truth)

    accuracy = accuracy_score(truths, preds)
    print(f"Accuracy: {accuracy:.2f}")

    precision = precision_score(truths, preds, average=None)
    recall = recall_score(truths, preds, average=None)
    f1 = f1_score(truths, preds, average=None)

    for cls in range(len(precision)):
        print(f"Class {cls} - Precision: {precision[cls]:.2f}, Recall: {recall[cls]:.2f}, F1-Score: {f1[cls]:.2f}")

    macro_precision = precision_score(truths, preds, average='macro')
    macro_recall = recall_score(truths, preds, average='macro')
    macro_f1 = f1_score(truths, preds, average='macro')

    print(f"\nMacro-Average Precision: {macro_precision:.2f}")
    print(f"Macro-Average Recall: {macro_recall:.2f}")
    print(f"Macro-Average F1-Score: {macro_f1:.2f}")

    micro_precision = precision_score(truths, preds, average='micro')
    micro_recall = recall_score(truths, preds, average='micro')
    micro_f1 = f1_score(truths, preds, average='micro')

    print(f"\nMicro-Average Precision: {micro_precision:.2f}")
    print(f"Micro-Average Recall: {micro_recall:.2f}")
    print(f"Micro-Average F1-Score: {micro_f1:.2f}")

if __name__ == "__main__":
    main()