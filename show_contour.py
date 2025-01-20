import cv2
import numpy as np
import os

def remove_near_white_pixels(image):
    if image.shape[2] == 3:  # RGB
        b, g, r = cv2.split(image)
        alpha = np.ones(b.shape, dtype=np.uint8) * 255  # Alpha kanalı
        image = cv2.merge((b, g, r, alpha))  # RGBA formatına dönüştür

    lower_white = np.array([200, 200, 200, 0])  # Beyazın alt sınırı
    upper_white = np.array([255, 255, 255, 255])  # Beyazın üst sınırı
    mask = cv2.inRange(image, lower_white, upper_white)
    image[mask > 0] = [0, 0, 0, 0]  # Beyaz pikselleri şeffaf yap
    return image

def detect_and_draw_contours(image):
    if image.shape[2] == 4:  # RGBA
        alpha_channel = image[:, :, 3]
    else:
        alpha_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0, 255), 1)  # RGBA için yeşil çizim
    return image

def draw_bounding_polygon_and_crop(processed_image, original_image, input_path):
    if processed_image.shape[2] == 4:  # RGBA
        alpha_channel = processed_image[:, :, 3]
    else:
        alpha_channel = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # İşlenmiş görüntüde konturları bul
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Tüm konturları birleştir
        all_contours = np.vstack(contours)
        # Bounding polygon'u hesapla
        bounding_polygon = cv2.convexHull(all_contours)
        
        # Bounding polygon'u çiz ve kaydet (artık step 4)
        polygon_image = original_image.copy()
        cv2.drawContours(polygon_image, [bounding_polygon], -1, (0, 255, 0, 255), 2)
        steps_dir = os.path.join(os.path.dirname(input_path), "steps")
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(steps_dir, f"{base_filename}_step4_bounding_polygon.png"), polygon_image)
        
        # Kırpma işlemi (artık step 5)
        x, y, w, h = cv2.boundingRect(bounding_polygon)
        cropped_original = original_image[y:y+h, x:x+w].copy()
        cv2.imwrite(os.path.join(steps_dir, f"{base_filename}_step5_cropped.png"), cropped_original)
        
        # Geri kalan işlemler...
        x, y, w, h = cv2.boundingRect(bounding_polygon)
        cropped_original = original_image[y:y+h, x:x+w].copy()
        
        # Orijinal ve kırpılmış görüntüleri yan yana koyma
        orig_h, orig_w = original_image.shape[:2]
        crop_h, crop_w = cropped_original.shape[:2]
        
        # En büyük boyutları bul
        max_height = max(orig_h, crop_h)
        max_width = max(orig_w, crop_w)
        
        # Çerçeve kalınlığı
        border = 2
        
        # Orijinal görüntü için kırmızı arkaplan oluştur (çerçeve için ekstra alan ekle)
        orig_with_red = np.zeros((max_height + 2*border, max_width + 2*border, 3), dtype=np.uint8)
        orig_with_red[:, :] = [0, 0, 255]  # Kırmızı arkaplan
        
        # Siyah çerçeve çiz
        orig_with_red[0:border, :] = [0, 0, 0]  # Üst çerçeve
        orig_with_red[-border:, :] = [0, 0, 0]  # Alt çerçeve
        orig_with_red[:, 0:border] = [0, 0, 0]  # Sol çerçeve
        orig_with_red[:, -border:] = [0, 0, 0]  # Sağ çerçeve
        
        # Orijinal görüntüyü kırmızı arkaplan üzerine yerleştir
        orig_with_red[border:orig_h+border, border:orig_w+border] = original_image[:, :, :3] if original_image.shape[2] == 4 else original_image
        
        # Kırpılmış görüntü için kırmızı arkaplan oluştur (çerçeve için ekstra alan ekle)
        crop_with_red = np.zeros((max_height + 2*border, max_width + 2*border, 3), dtype=np.uint8)
        crop_with_red[:, :] = [0, 0, 255]  # Kırmızı arkaplan
        
        # Siyah çerçeve çiz
        crop_with_red[0:border, :] = [0, 0, 0]  # Üst çerçeve
        crop_with_red[-border:, :] = [0, 0, 0]  # Alt çerçeve
        crop_with_red[:, 0:border] = [0, 0, 0]  # Sol çerçeve
        crop_with_red[:, -border:] = [0, 0, 0]  # Sağ çerçeve
        
        # Kırpılmış görüntüyü kırmızı arkaplan üzerine yerleştir
        crop_with_red[border:crop_h+border, border:crop_w+border] = cropped_original[:, :, :3] if cropped_original.shape[2] == 4 else cropped_original
        
        # Yan yana birleştir
        combined_image = np.hstack((orig_with_red, crop_with_red))
        
        # Combined klasörü oluştur
        output_dir = os.path.join(os.path.dirname(input_path), "combined")
        os.makedirs(output_dir, exist_ok=True)
        
        # Birleştirilmiş görüntüyü combined klasörüne kaydet
        output_path = os.path.join(output_dir, os.path.basename(input_path).replace(".png", "_combined.png"))
        cv2.imwrite(output_path, combined_image)
        print(f"Combined image saved as: {output_path}")
        
        return cropped_original, combined_image
    
    return None, None


import cv2
import numpy as np

def remove_small_non_intersecting_contours(image):
    height, width = image.shape[:2]
    left_limit = int(width * 0.15)
    top_limit = int(height * 0.15)

    if image.shape[2] == 4:  # RGBA
        alpha_channel = image[:, :, 3]
        color_channels = image[:, :, :3]
    else:  # RGB
        alpha_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_channels = image

    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Sol üst köşe kontrolü ve alan kontrolü
        if x + w <= left_limit and y + h <= top_limit and area < 4500:
            # Kontur için maske oluştur
            mask = np.zeros_like(alpha_channel)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Kontur bölgesini al
            roi = color_channels[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            
            # Kontür içindeki pikselleri kontrol et
            for i in range(roi.shape[0]):
                for j in range(roi.shape[1]):
                    pixel = roi[i, j]
                    # Koyu renk kontrolü
                    if np.all(pixel < 50):  # Koyu renk
                        continue  # Koyu renkli pikselleri koru
                    else:
                        # Koyu renk değilse, pikseli sil
                        image[y + i, x + j] = (0, 0, 0, 0)  # Şeffaf yap
                        print(f"Kontur ({x}, {y}) içindeki açık renkli piksel siliniyor: {pixel}")

    return image


def process_image(image_path):
    # Orijinal görüntüyü oku
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        print(f"Error reading image: {image_path}")
        return None

    # Steps klasörü oluştur
    steps_dir = os.path.join(os.path.dirname(image_path), "steps")
    os.makedirs(steps_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # İşlenecek görüntü için kopya oluştur
    processed_image = original_image.copy()

    # Adım 1: Beyaza yakın pikselleri kaldır
    processed_image = remove_near_white_pixels(processed_image)
    cv2.imwrite(os.path.join(steps_dir, f"{base_filename}_step1_white_removed.png"), processed_image)

    # Adım 2: Konturları tespit et ve çiz
    processed_image = detect_and_draw_contours(processed_image)
    cv2.imwrite(os.path.join(steps_dir, f"{base_filename}_step2_contours.png"), processed_image)

    # Adım 3: Küçük kesişmeyen konturları kaldır
    processed_image = remove_small_non_intersecting_contours(processed_image)
    cv2.imwrite(os.path.join(steps_dir, f"{base_filename}_step3_small_contours_removed.png"), processed_image)
    
    # Adım 4 ve 5: Bounding polygon ve kırpma işlemi
    cropped_image, combined_image = draw_bounding_polygon_and_crop(processed_image, original_image, image_path)
    
    return cropped_image, combined_image

def process_all_png_in_folder(input_folder):
    # Klasördeki tüm PNG dosyalarını al
    png_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    
    if not png_files:
        print("Klasörde PNG dosyası bulunamadı.")
        return

    # Combined klasörü oluştur
    combined_folder = os.path.join(input_folder, "combined")
    os.makedirs(combined_folder, exist_ok=True)
    print(f"Created combined folder at: {combined_folder}")

    for png_file in png_files:
        input_path = os.path.join(input_folder, png_file)
        print(f"Processing: {input_path}")
        
        # İşlenmiş görüntüyü al
        cropped_image, combined_image = process_image(input_path)
        
        if cropped_image is not None:
            print(f"Successfully processed: {input_path}")


# Klasör yolu
input_folder = "C://Users//impark//Desktop//impark//deepcut//test//soru_kes_testler//lise-destek"
for alt_öge in os.listdir(input_folder):
    alt_oge_yolu = os.path.join(input_folder, alt_öge)
    if os.path.isdir(alt_oge_yolu):  # Eğer bu bir klasörse
        process_all_png_in_folder(alt_oge_yolu)
        print(alt_oge_yolu)

 