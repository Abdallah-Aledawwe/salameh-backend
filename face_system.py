import cv2
import numpy as np
import base64
from models import FaceScan
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session
from sqlalchemy import text
from PIL import Image, ImageOps
import io


# 🔥 تحميل المودل مرة وحدة فقط
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        print("🚀 Loading InsightFace ONCE...")
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]  # 👈 بدون GPU
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace Ready!")
    return face_app


class FaceEngine:
    def __init__(self):
        self.app = get_face_app()

    def get_encoding(self, img_rgb: np.ndarray) -> dict:
        try:
            faces = self.app.get(img_rgb)

            if len(faces) == 0:
                return {"status": "no_face"}
            if len(faces) > 1:
                return {"status": "multiple_faces"}

            return {
                "status": "ok",
                "encoding": faces[0].embedding.tolist()
            }
        except Exception as e:
            print(f"❌ Encoding Error: {e}")
            return {"status": "error"}

    def decode_base64(self, base64_str: str):
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            img_data = base64.b64decode(base64_str)

            img = Image.open(io.BytesIO(img_data))
            img = ImageOps.exif_transpose(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"❌ Decode Error: {e}")
            return None

    def find_match(self, img_rgb: np.ndarray, db: Session, threshold: float = 0.6) -> dict:
        result = self.get_encoding(img_rgb)
        if result["status"] != "ok":
            return {"status": result["status"]}

        encoding = result["encoding"]

        try:
            dist_col = FaceScan.encoding.cosine_distance(encoding).label("dist")
            result_row = (
                db.query(FaceScan, dist_col)
                .filter(FaceScan.encoding.is_not(None))
                .order_by("dist")
                .first()
            )

            if not result_row:
                return {"status": "empty_db"}

            row, distance = result_row
            similarity = 1 - float(distance)

            if similarity < threshold:
                return {"status": "unknown"}

            accuracy = round(similarity * 100, 1)

            if row.childid:
                from models import Child
                child = db.query(Child).filter(Child.childid == row.childid).first()
                if child:
                    return {
                        "status": "found",
                        "type": "child",
                        "name": child.fullname,
                        "identity_id": child.nationalityid,
                        "accuracy": accuracy
                    }

            elif row.userid:
                from models import User
                user = db.query(User).filter(User.userid == row.userid).first()
                if user:
                    return {
                        "status": "found",
                        "type": "user",
                        "name": user.fullname,
                        "identity_id": user.email,
                        "accuracy": accuracy
                    }

            return {"status": "unknown"}

        except Exception as e:
            print(f"❌ Search Error: {e}")
            return {"status": "error"}

    def is_duplicate(
        self,
        encoding: list,
        db: Session,
        threshold: float = 0.45,
        exclude_userid: int = None,
        exclude_childid: int = None
    ) -> bool:
        try:
            dist_col = FaceScan.encoding.cosine_distance(encoding).label("dist")
            query = db.query(dist_col).filter(FaceScan.encoding.is_not(None))

            if exclude_userid:
                query = query.filter(FaceScan.userid.is_distinct_from(exclude_userid))
            if exclude_childid:
                query = query.filter(FaceScan.childid.is_distinct_from(exclude_childid))

            min_dist_row = query.order_by("dist").first()
            if not min_dist_row:
                return False

            similarity = 1 - float(min_dist_row[0])
            print(f"🔍 Duplicate check — highest similarity: {similarity:.3f}")
            return similarity >= threshold

        except Exception as e:
            print(f"❌ Duplicate Check Error: {e}")
            return False


# 🔥 instance واحد فقط
face_engine = FaceEngine()