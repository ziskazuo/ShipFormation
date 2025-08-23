import os


def merge_txt_files_in_folder(input_folder: str, output_file: str):
    """
    读取 input_folder 文件夹中的所有 .txt 文件，
    合并它们的内容到一个 output_file 中，每个文件内容之间用空行分隔。

    :param input_folder: 输入的文件夹路径，包含多个 .txt 文件
    :param output_file: 输出的合并后的 .txt 文件路径
    """
    # 获取文件夹下所有 .txt 文件
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # 按文件名排序（可选，如果需要按某种顺序合并）
    txt_files.sort()

    if not txt_files:
        print(f"⚠️  在文件夹 '{input_folder}' 中没有找到 .txt 文件。")
        return

    print(f"📂 找到 {len(txt_files)} 个 .txt 文件，开始合并...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, filename in enumerate(txt_files):
            filepath = os.path.join(input_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)

                    # 如果不是最后一个文件，则写入一个空行分隔
                    if i < len(txt_files) - 1:
                        outfile.write('\n\n')  # 空行分隔

                    print(f"✅ 已合并: {filename}")
            except Exception as e:
                print(f"❌ 读取文件 {filename} 时出错: {e}")

    print(f"🎉 合并完成！结果已保存到: {output_file}")


# ==========================
# 使用示例（请修改为你的文件夹路径）
# ==========================

if __name__ == "__main__":
    # 📁 请替换为你的 txt 文件所在的文件夹路径
    input_folder_path = "./Data/data"  # 例如：当前目录下的 txt_files 文件夹

    # 📄 请替换为你想要保存的合并后的输出文件路径
    output_merged_file = "./Data/data.txt"  # 合并后的总文件

    # 调用函数进行合并
    merge_txt_files_in_folder(input_folder_path, output_merged_file)