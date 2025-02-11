# [2025-01-17 09:58:10 AM-eval.py-INFO:Testing on CVC-300 dataset...]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Mean val dice: 0.9425003111362458]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Mean val gd: 0.9425002982219061]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Mean val iou: 0.8941938733061154]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Testing on CVC-ClinicDB dataset...]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Mean val dice: 0.9478659591367168]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Mean val gd: 0.947865960098082]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Mean val iou: 0.9031791023669704]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Testing on CVC-ColonDB dataset...]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Mean val dice: 0.9235947792467318]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Mean val gd: 0.9235947805015664]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Mean val iou: 0.8644482552220947]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Testing on ETIS-LaribPolypDB dataset...]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Mean val dice: 0.9444629169848501]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Mean val gd: 0.9444629188094821]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Mean val iou: 0.8965879900723087]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Testing on Kvasir dataset...]
# [2025-01-17 10:01:01 AM-eval.py-INFO:Mean val dice: 0.9599819499254226]
# [2025-01-17 10:01:01 AM-eval.py-INFO:Mean val gd: 0.9599819546937942]
# [2025-01-17 10:01:01 AM-eval.py-INFO:Mean val iou: 0.9243155312538147]

# def extract(text):
#     lines = text.split('\n')
#     print(lines)
#     dataset = None
    
#     for line in lines:
def get_intervals(length,num):
        res = []
        for i in range(num):
            res.append(round(length / num ))
        return res
import torch
if __name__ == '__main__':
#     extract("""[2025-01-17 09:58:10 AM-eval.py-INFO:Testing on CVC-300 dataset...]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Mean val dice: 0.9425003111362458]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Mean val gd: 0.9425002982219061]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Mean val iou: 0.8941938733061154]
# [2025-01-17 09:58:23 AM-eval.py-INFO:Testing on CVC-ClinicDB dataset...]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Mean val dice: 0.9478659591367168]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Mean val gd: 0.947865960098082]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Mean val iou: 0.9031791023669704]
# [2025-01-17 09:58:35 AM-eval.py-INFO:Testing on CVC-ColonDB dataset...]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Mean val dice: 0.9235947792467318]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Mean val gd: 0.9235947805015664]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Mean val iou: 0.8644482552220947]
# [2025-01-17 09:59:46 AM-eval.py-INFO:Testing on ETIS-LaribPolypDB dataset...]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Mean val dice: 0.9444629169848501]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Mean val gd: 0.9444629188094821]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Mean val iou: 0.8965879900723087]
# [2025-01-17 10:00:39 AM-eval.py-INFO:Testing on Kvasir dataset...]
# [2025-01-17 10:01:01 AM-eval.py-INFO:Mean val dice: 0.9599819499254226]
# [2025-01-17 10:01:01 AM-eval.py-INFO:Mean val gd: 0.9599819546937942]
# [2025-01-17 10:01:01 AM-eval.py-INFO:Mean val iou: 0.9243155312538147]""")
    a = torch.ones((256))
    print(get_intervals(256,4))
    print(a.split(get_intervals(256,4)))