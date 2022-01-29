//
//  ViewController.swift
//  CoreMLDemo
//
//  Created by Sai Kambampati on 14/6/2017.
//  Copyright © 2017 AppCoda. All rights reserved.
//

import UIKit
import CoreML
import Vision
import VideoToolbox

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!//選中圖片
    @IBOutlet weak var classifier: UILabel!//Label
    @IBOutlet weak var processTime: UILabel!
    
    var model: ResNet50!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    override func viewWillAppear(_ animated: Bool) {//載入view之後（接在viewDidLoad後)，畫面顯示前
        model = ResNet50()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func camera(_ sender: Any) {
        
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            return
        }
        
        let cameraPicker = UIImagePickerController()//物件
        cameraPicker.delegate = self//將代理指向類本身
        cameraPicker.sourceType = .camera//資料來源類型設為相機
        cameraPicker.allowsEditing = false
        
        present(cameraPicker, animated: true)//開啟相機
    }
    
    @IBAction func openLibrary(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.allowsEditing = false
        picker.delegate = self//將代理指向類本身
        picker.sourceType = .photoLibrary//資料來源類型設為相簿
        present(picker, animated: true)//開啟相簿
    }

    func predictUsingVision(image: UIImage) {
      guard let visionModel = try? VNCoreMLModel(for: model.model) else {
        fatalError("Someone did a baddie")
      }

      let request = VNCoreMLRequest(model: visionModel) { request, error in
        if let observations = request.results as? [VNClassificationObservation] {

          // The observations appear to be sorted by confidence already, so we
          // take the top 4 and map them to an array of (String, Double) tuples.
          let top4 = observations.prefix(through: 3)
                                 .map { ($0.identifier, Double($0.confidence)) }
          self.show(results: top4)
        }
      }

      request.imageCropAndScaleOption = .centerCrop

      let handler = VNImageRequestHandler(cgImage: image.cgImage!)
      try? handler.perform([request])
    }
    
    typealias Prediction = (String, Double)

    func show(results: [Prediction]) {
      var s: [String] = []
      for (i, pred) in results.enumerated() {
        s.append(String(format: "%d: %@ (%3.2f%%)", i + 1, pred.0, pred.1 * 100))
      }
        classifier.text = s.joined(separator: "\n\n")
    }
    
    func top(_ k: Int, _ prob: [String: Double]) -> [Prediction] {
      precondition(k <= prob.count)

      return Array(prob.map { x in (x.key, x.value) }
                       .sorted(by: { a, b -> Bool in a.1 > b.1 })
                       .prefix(through: k - 1))
    }
}

extension ViewController: UIImagePickerControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)//關閉相簿
    }
    //在相簿中選擇相片後，代理自動呼叫，用來獲取選中相片
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        picker.dismiss(animated: true)
        classifier.text = "Analyzing Image..."
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        } //獲取被選擇的相片
        
        /* UIGraphicsBeginImageContextWithOptions
         * 參數一: 指定創建出来的bitmap的大小
         * 參數二: 设置透明YES代表透明，NO代表不透明
         * 參數三: 代表缩放,0代表不缩放
         */
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 299, height: 299), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: 299, height: 299))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        /*
         把newImage轉換為CVPixelBuffer，CVPixelBuffer是一個將像素(pixcel)存在主記憶體的圖像緩衝器
         */
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(newImage.size.width), Int(newImage.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        /*
         取得這個圖像裡的像素並轉換為裝置的RGB色彩，接著把這資料作成CGContext
         */
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(newImage.size.width), height: Int(newImage.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) //3
        
        context?.translateBy(x: 0, y: newImage.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        /*
         完成新圖像的繪製並把舊圖像的資料移除，將newImage指定給imageView.image
         */
        UIGraphicsPushContext(context!)
        newImage.draw(in: CGRect(x: 0, y: 0, width: newImage.size.width, height: newImage.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        imageView.image = newImage
        
        // Core ML
        guard let prediction = try? model.prediction(image: pixelBuffer!) else {
            return
        }
        //classifier.text = "I think this is a \(prediction.classLabel)."
        
        let startTime = CFAbsoluteTimeGetCurrent()
        predictUsingVision(image: newImage)
        let endTime = CFAbsoluteTimeGetCurrent()
        let time = String(format: "模型辨識時間: %f 秒", (endTime - startTime))
        processTime.text = time
    }
}
