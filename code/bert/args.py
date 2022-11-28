import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/workspace/kch_dkt/data",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    # 최대 시퀀스 길이 설정
    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=6, type=int, help="number of workers") # 1

    # 모델
    parser.add_argument(
        "--hidden_dim", default=128, type=int, help="hidden dimension size"
    )   # 64
    parser.add_argument("--n_layers", default=6, type=int, help="number of layers") # 2
    parser.add_argument("--n_heads", default=32, type=int, help="number of heads")   # 2
    parser.add_argument("--drop_out", default=0.3, type=float, help="drop out rate")    # 0.2

    # 훈련
    parser.add_argument("--n_epochs", default=30, type=int, help="number of epochs")    # 20
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")   # 64
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")   # 0.0001
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=100, type=int, help="print log per n steps"
    )   # 50

    ### 중요 ###
    parser.add_argument("--model", default="bert", type=str, help="model type") # lstm, lstmattn
    parser.add_argument("--optimizer", default="adamW", type=str, help="optimizer type")    # adam
    parser.add_argument(
        "--scheduler", default="linear_warmup", type=str, help="scheduler type" # plateau
    )

    args = parser.parse_args()

    return args
