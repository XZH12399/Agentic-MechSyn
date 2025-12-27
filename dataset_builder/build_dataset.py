import os
import json
import glob
from augmentor import DataAugmentor


def main():
    # === 配置 ===
    SEEDS_DIR = "seeds"  # 相对路径
    OUTPUT_FILE = "output/mech_graph_dataset_v1.jsonl"
    VARIANTS_PER_SEED = 100  # 每个种子生成 100 个变体

    # 模拟优化器配置
    class OptConfig:
        learning_rate = 0.01
        max_iterations = 1000  # 修复步数给够

    augmentor = DataAugmentor(OptConfig())

    # 准备输出文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)  # 清空旧数据

    # 遍历种子生成
    seed_files = glob.glob(os.path.join(SEEDS_DIR, "*.json"))
    total = 0

    for seed_path in seed_files:
        print(f"\n🚀 Start processing: {seed_path}")
        try:
            new_data = augmentor.augment_from_seed(seed_path, num_variants=VARIANTS_PER_SEED)

            # 写入文件
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for record in new_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += len(new_data)
            print(f"✅ Saved {len(new_data)} variants.")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎉 All done! Total generated: {total}")


if __name__ == "__main__":
    main()