import os
import shutil

def copy_combined_images(parent_folder, output_folder):
    """
    Belirtilen parent_folder içinde "combined" kelimesini içeren tüm PNG dosyalarını
    output_folder'a kopyalar. Output klasöründe sadece PNG dosyaları olacak.
    
    Args:
        parent_folder (str): Ana dizin yolu.
        output_folder (str): Hedef dizin yolu.
    """
    # Eğer output klasörü varsa önce sil
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Yeni ve boş output klasörünü oluştur
    os.makedirs(output_folder)
    
    # Parent folder içindeki tüm alt dizinlerde gez
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            # Dosya adında "combined" geçen ve .png ile biten dosyaları bul
            if "combined" in file.lower() and file.lower().endswith('.png'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(output_folder, file)
                
                # Dosyayı kopyala
                shutil.copy2(source_path, target_path)
                print(f"Kopyalandı: {source_path} -> {target_path}")

if __name__ == "__main__":
    parent_folder = "C://Users//impark//Desktop//impark//deepcut//test//soru_kes_testler"  # Ana klasör yolu
    output_folder =  "C://Users//impark//Desktop//impark//deepcut//test//soru_kes_testler_birarada"  # Çıktı klasörü
    copy_combined_images(parent_folder, output_folder)
