import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu") # cpu

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
    parser.add_argument("--max_seq_len", default=15, type=int, help="max sequence length")  # [15], 20, 27
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers") # 1, 6, [8]

    # 모델
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden dimension size")   # 64, [128], 256, 512, 768
    parser.add_argument("--n_layers", default=4, type=int, help="number of layers") # 1, 2, [4], 6, 8
    parser.add_argument("--n_heads", default=4, type=int, help="number of heads")   # 2, [4], 16, 32, 64
    parser.add_argument("--drop_out", default=0.3, type=float, help="drop out rate")    # 0.1, 0.2, [0.3], 0.4

    # bert 모델 구조
    parser.add_argument("--intermediate_size", default=3072, type=int, help="number of intermediate layer")    # 1536, [3072], 6144
    parser.add_argument("--hidden_act", default="gelu", type=str, help="non-linear activation function")    # ["gelu"], "relu", "silu", "gelu_new"
    parser.add_argument("--layer_norm_eps", default=1e-10, type=float, help="epsilon used by normalization")    # 1e-9, [1e-10], 1e-12, 1e-13

    # 훈련
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")   # 64, 128, 256, 512
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")   # 0.0001, 0.0002, 0.0003
    parser.add_argument("--clip_grad", default=32, type=int, help="clip grad")  # 10, 20, 32, 64
    
    parser.add_argument("--n_epochs", default=50, type=int, help="number of epochs")    # 20, 30, 50
    parser.add_argument("--patience", default=10, type=int, help="for early stopping")
    parser.add_argument("--log_steps", default=100, type=int, help="print log per n steps")

    ### 중요 ###
    parser.add_argument("--model", default="bert", type=str, help="model type") # lstm, lstmattn
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")    # [adam], adamW
    parser.add_argument("--scheduler", default="linear_warmup", type=str, help="scheduler type")    # plateau, [linear_warmup]

    args = parser.parse_args()

    return args
