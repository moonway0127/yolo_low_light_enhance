import os
import glob
import time
from test_infer import TestInfer

def find_working_image():
    """循环测试所有图片，直到找到增强模型能检测到的图片"""
    
    # 获取所有高清图片
    image_files = sorted(glob.glob('sample_data/high_res/*.jpg'))
    
    if not image_files:
        print("没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("开始循环测试...\n")
    
    # 初始化测试器
    tester = TestInfer(
        model_path='yolo26n.pt',
        enhanced_model_path='runs/train/test_train_1773657540/best.pt'
    )
    
    total_tested = 0
    success_count = 0
    
    for image_path in image_files:
        total_tested += 1
        
        try:
            print(f"\n[{total_tested}] 测试：{os.path.basename(image_path)}")
            
            # 记录总时间
            start_time = time.time()
            
            # 运行测试
            result = tester.run_test(image_path)
            
            total_time = time.time() - start_time
            
            native_count = result['native_count']
            enhanced_count = result['enhanced_count']
            native_time = result.get('native_time', 0)
            enhanced_time = result.get('enhanced_time', 0)
            
            print(f"  原生 YOLO: {native_count} 个目标，用时：{native_time*1000:.1f}ms")
            print(f"  增强 YOLO: {enhanced_count} 个目标，用时：{enhanced_time*1000:.1f}ms")
            print(f"  总耗时：{total_time*1000:.1f}ms")
            
            if enhanced_count > 0:
                print(f"\n✅ 成功！找到增强模型能检测的图片：{os.path.basename(image_path)}")
                print(f"  检测目标数：{enhanced_count}")
                print(f"  总测试图片数：{total_tested}")
                print(f"  成功率：{success_count/total_tested*100:.1f}%")
                
                # 打印详细时间
                if 'native_time' in result:
                    print(f"  原生 YOLO 时间：{result['native_time']*1000:.1f}ms")
                if 'enhanced_time' in result:
                    print(f"  增强 YOLO 时间：{result['enhanced_time']*1000:.1f}ms")
                
                return image_path
            
            success_count += 1 if enhanced_count > 0 else 0
            
        except Exception as e:
            print(f"  测试失败：{e}")
            continue
    
    print(f"\n❌ 测试完所有 {total_tested} 张图片，没有找到增强模型能检测的图片")
    return None

if __name__ == '__main__':
    find_working_image()
