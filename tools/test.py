import torch
from torch_geometric.loader import DataLoader
from gravit.models import build_model
from gravit.datasets import GraphDataset

def predict(cfg, data_path):
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型并加载权重
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(os.path.join(cfg['root_result'], cfg['exp_name'], 'ckpt_best.pt')))
    model.to(device)
    model.eval()
    
    # 准备数据加载器
    test_loader = DataLoader(GraphDataset(data_path), batch_size=cfg['batch_size'], shuffle=False)
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.cpu().numpy())
    
    # 将所有预测结果整合
    predictions = np.concatenate(predictions)
    return predictions

if __name__ == "__main__":
    args = get_args()
    cfg = get_cfg(args)
    
    # 设置数据路径
    test_data_path = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}/test')
    
    # 获取预测结果
    predictions = predict(cfg, test_data_path)
    
    # 输出预测结果
    print(predictions)
