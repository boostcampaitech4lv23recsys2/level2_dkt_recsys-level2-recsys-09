class Config:
    
    num_layers = 2
    num_heads = 4
    seq_len = 100
    
    model_dim = 64
    max_len = 1000
    
    num_question = 9454
    num_test = 1537

    dropout = 0.1
    epochs = 100
    batch_size = 80
    learning_rate = 0.1
    warmup = 4000
    
    train_path = '/opt/ml/input/DKT/data/train_data.csv'
    test_path = '/opt/ml/input/DKT/data/test_data.csv'
    sub_path = '/opt/ml/input/DKT/data/sample_submission.csv'
    
    user_wandb = True
    wandb_kwargs = dict(project="dkt-saint")
    
