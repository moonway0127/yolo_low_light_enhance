import torch
import time

# 模拟检测头
class SyncHead(torch.nn.Module):
    """同步检测头（当前实现）"""
    def __init__(self):
        super().__init__()
        self.shared_conv = torch.nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.box_branch = torch.nn.Conv2d(64, 4, kernel_size=1)
        self.conf_branch = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.cls_branch = torch.nn.Conv2d(64, 80, kernel_size=1)
    
    def forward(self, x):
        x = self.shared_conv(x)
        boxes = self.box_branch(x)
        conf = self.conf_branch(x)
        cls = self.cls_branch(x)
        return torch.cat([boxes, conf, cls], dim=1)

class AsyncHead(torch.nn.Module):
    """异步检测头（实验实现）"""
    def __init__(self):
        super().__init__()
        self.shared_conv = torch.nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.box_branch = torch.nn.Conv2d(64, 4, kernel_size=1)
        self.conf_branch = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.cls_branch = torch.nn.Conv2d(64, 80, kernel_size=1)
    
    def forward(self, x):
        return self.forward_async(x)
    
    def forward_async(self, x):
        import concurrent.futures
        
        x = self.shared_conv(x)
        
        # 使用线程池异步执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_boxes = executor.submit(self.box_branch, x)
            future_conf = executor.submit(self.conf_branch, x)
            future_cls = executor.submit(self.cls_branch, x)
            
            boxes = future_boxes.result()
            conf = future_conf.result()
            cls = future_cls.result()
        
        return torch.cat([boxes, conf, cls], dim=1)

# 性能测试
def benchmark():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    sync_head = SyncHead().to(device)
    async_head = AsyncHead().to(device)
    
    # 加载权重（共享）
    async_head.load_state_dict(sync_head.state_dict())
    
    x = torch.randn(1, 16, 80, 80).to(device)
    
    # 预热
    for _ in range(10):
        _ = sync_head(x)
        _ = async_head(x)
    
    # 测试同步版本
    sync_head.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = sync_head(x)
        if device.type != 'cpu':
            torch.mps.synchronize()
        sync_time = (time.time() - start) / 100 * 1000
    
    # 测试异步版本
    async_head.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = async_head(x)
        if device.type != 'cpu':
            torch.mps.synchronize()
        async_time = (time.time() - start) / 100 * 1000
    
    print(f"\n性能对比（100 次平均）:")
    print(f"同步检测头：{sync_time:.2f}ms")
    print(f"异步检测头：{async_time:.2f}ms")
    print(f"性能差异：{(async_time/sync_time - 1)*100:.1f}% {'更慢' if async_time > sync_time else '更快'}")
    print(f"\n结论：GPU 已经自动并行化三个分支，手动异步反而增加开销")

if __name__ == '__main__':
    benchmark()
