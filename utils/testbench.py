import multiprocessing as mp
import sys, os
import pandas as pd
import train


def run_single_benchmark(user:int, benchmark_file_path):
    """"
    workflow goes like this:
        A model will be trained for a specified user using cross validation, the results will be written to an intermediate file?
    """
    kwargs = {
        "mode": train.LoadMode.ONE_HOT,
        "epochs_num": 500,
        "rows_per_example": 50,
        "hidden_dim": 128,
        "user_id": user,
        "num_layers": 2,
        "use_fc_before": False
    }

    args = pd.DataFrame([kwargs])

    l, model_summary_df = train.train_with_crossvalidation(
        "../keystroke_data.sqlite", model_path='../models/test.pth', test_train_split=0.2, positive_negative_ratio=1, offset=1,
            **kwargs)


    # aggregate the precission and recall
    precision = sum([i[0] for i in l])/len(l)
    recall = sum([i[1] for i in l])/len(l)
    df2 = pd.DataFrame([(precision, recall)], columns=["precision", "recall"])

    
    full_report_df = pd.concat([args, model_summary_df, df2], axis=1)

    if os.path.exists(benchmark_file_path):
        full_report_df.to_csv(benchmark_file_path,  mode='a', header=False, index=False)
    else:
        full_report_df.to_csv(benchmark_file_path, header='column_names', index=False)


if __name__ == '__main__':
    user_num = sys.argv[1]
    run_single_benchmark(int(user_num), f"key_presses_{user_num}.csv")