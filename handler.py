import os, json, cv2, numpy as np, boto3
from core import Segmenter

s3   = boto3.client("s3")
cols = int(os.environ.get("ARRAY_COLS","4"))
rows = int(os.environ.get("ARRAY_ROWS","5"))
MODEL_PATH="/var/task/u2net_multiclass.pt"

_seg = None
def get_seg():
    global _seg
    if _seg is None:
        _seg = Segmenter(MODEL_PATH, device="cpu", img_size=512, grid=(cols,rows))
    return _seg

def lambda_handler(event, context):
    for rec in event.get("Records", []):
        bkt = rec["s3"]["bucket"]["name"]
        key = rec["s3"]["object"]["key"]
        if not key.lower().endswith((".jpg",".jpeg",".png")): continue

        obj = s3.get_object(Bucket=bkt, Key=key)
        data= np.frombuffer(obj["Body"].read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None: continue

        seg = get_seg()
        _, fg = seg.predict_mask(bgr)
        rgba  = seg.cutout_rgba(bgr, fg)
        rb, rr, rg = seg.rectify(bgr, fg, out_width=1500, pad=20)

        base,_ = os.path.splitext(os.path.basename(key))
        def put(arr,suf):
            ok,enc = cv2.imencode(".png", arr)
            s3.put_object(Bucket=bkt, Key=f"outputs/{base}{suf}.png", Body=enc.tobytes(), ContentType="image/png")
        put(fg,   "_fgmask")
        put(rgba, "_bg_removed")
        if rr is not None:
            put(rr, "_rect_rgba"); put(rb, "_rect_bgr"); put(rg, "_rect_grid")
    return {"statusCode":200, "body": json.dumps("OK")}
