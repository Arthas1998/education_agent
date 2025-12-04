import os
import threading
from typing import Dict, Optional, List
from PIL import Image
import fitz  # PyMuPDF
import io
import base64


class CourseMeta:
    def __init__(self, course_id: str, title: str, path: str, total_pages: int):
        self.id = course_id
        self.title = title
        self.path = path
        self.total_pages = total_pages


class PDFManager:
    """Scan PDFs in a directory and provide per-page image access.

    - Scans pdf_dir for files named like "<id>_<title>.pdf".
    - Stores images under img_dir/<id>_<title>/<page>.jpg
    """

    def __init__(self, pdf_dir: str, img_dir: str, static_url_prefix: str = "/static"):
        self.pdf_dir = pdf_dir
        self.img_dir = img_dir
        self.static_url_prefix = static_url_prefix
        self._lock = threading.Lock()
        self._converting = set()  # track in-progress conversions by course_id
        self.courses: Dict[str, CourseMeta] = {}
        self.scan_pdfs()

    def scan_pdfs(self):
        """Scan the pdf_dir and populate self.courses."""
        if not os.path.exists(self.pdf_dir):
            return
        for fn in sorted(os.listdir(self.pdf_dir)):
            if not fn.lower().endswith('.pdf'):
                continue
            name = fn[:-4]
            # try parse id_title or fallback to whole name as id
            if "_" in name:
                course_id, title = name.split("_", 1)
            else:
                course_id = name
                title = name
            path = os.path.join(self.pdf_dir, fn)
            total = self._get_pdf_page_count(path)
            meta = CourseMeta(course_id=course_id, title=title, path=path, total_pages=total)
            self.courses[course_id] = meta
            # ensure image dir exists
            os.makedirs(self.get_image_dir(course_id, title), exist_ok=True)

    def _get_pdf_page_count(self, path: str) -> int:
        try:
            doc = fitz.open(path)
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    def get_image_dir(self, course_id: str, title: str) -> str:
        return os.path.join(self.img_dir, f"{course_id}_{title}")

    def list_courses(self) -> List[Dict]:
        out = []
        for c in self.courses.values():
            cover_url = None
            # if cover exists
            cover_path = os.path.join(self.get_image_dir(c.id, c.title), "1.jpg")
            if os.path.exists(cover_path):
                cover_url = f"{self.static_url_prefix}/{c.id}_{c.title}/1.jpg"
            out.append({
                "id": c.id,
                "title": c.title,
                "total_pages": c.total_pages,
                "cover_url": cover_url,
            })
        return out

    def get_page_image_path(self, course_id: str, page: int) -> Optional[str]:
        meta = self.courses.get(course_id)
        if not meta:
            return None
        img_path = os.path.join(self.get_image_dir(meta.id, meta.title), f"{page}.jpg")
        if os.path.exists(img_path):
            return img_path
        return None

    def get_page_image_url(self, course_id: str, page: int) -> Optional[str]:
        meta = self.courses.get(course_id)
        if not meta:
            return None
        img_path = os.path.join(self.get_image_dir(meta.id, meta.title), f"{page}.jpg")
        if os.path.exists(img_path):
            return f"{self.static_url_prefix}/{meta.id}_{meta.title}/{page}.jpg"
        return None

    def ensure_images(self, course_id: str):
        """Synchronous conversion of entire PDF to images (low priority task)."""
        meta = self.courses.get(course_id)
        if not meta:
            raise FileNotFoundError("Course not found")
        outdir = self.get_image_dir(meta.id, meta.title)
        # convert using PyMuPDF
        doc = fitz.open(meta.path)
        for i in range(len(doc)):
            page = doc[i]
            mat = fitz.Matrix(1.25, 1.25)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes(output="png")
            # save as jpg
            im = Image.open(io.BytesIO(img_bytes))
            jpg_path = os.path.join(outdir, f"{i+1}.jpg")
            im.convert("RGB").save(jpg_path, format="JPEG", quality=75)
        doc.close()

    def convert_async(self, course_id: str):
        """Start background conversion for a course if not already running."""
        with self._lock:
            if course_id in self._converting:
                return
            self._converting.add(course_id)

        def _worker():
            try:
                self.ensure_images(course_id)
            except Exception:
                # log in real app
                pass
            finally:
                with self._lock:
                    self._converting.discard(course_id)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def get_page_image_base64(self, course_id: str, page: int, max_width: int = 800) -> Optional[str]:
        """Return base64-encoded JPEG bytes for inclusion in prompts (resized to max_width)."""
        path = self.get_page_image_path(course_id, page)
        if not path:
            return None
        with Image.open(path) as im:
            w, h = im.size
            if w > max_width:
                new_h = int(max_width * h / w)
                im = im.resize((max_width, new_h))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"


if __name__ == "__main__":
    # Full conversion test: convert all PDFs found under data/pdf to per-page JPGs
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base, "data", "pdf")
    img_dir = os.path.join(base, "backend", "static")

    pm = PDFManager(pdf_dir=pdf_dir, img_dir=img_dir)

    courses = pm.list_courses()
    if not courses:
        print("No courses found in:", pdf_dir)
    else:
        print(f"Found {len(courses)} course(s). Starting synchronous conversion...")

    for course in courses:
        cid = course.get("id")
        title = course.get("title")
        total_pages = course.get("total_pages")
        print("\n---")
        print(f"Processing course: {cid} - {title} (total pages: {total_pages})")
        try:
            pm.ensure_images(cid)
        except Exception as e:
            print(f"Error converting course {cid}: {e}")
            continue

        image_dir = pm.get_image_dir(cid, title)
        if not os.path.exists(image_dir):
            print(f"Image dir not found after conversion: {image_dir}")
            continue

        jpgs = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith('.jpg')]
        jpg_count = len(jpgs)
        cover_path = os.path.join(image_dir, '1.jpg') if os.path.exists(os.path.join(image_dir, '1.jpg')) else None

        # get base64 preview for page 1 if exists
        b64_len = 0
        if cover_path:
            try:
                b64 = pm.get_page_image_base64(cid, 1)
                if b64:
                    b64_len = len(b64)
            except Exception as e:
                print(f"Failed to read base64 for {cid} page 1: {e}")

        print(f"Converted -> saved_jpg_count={jpg_count}, cover_exists={bool(cover_path)}, cover_path={cover_path}, page1_base64_len={b64_len}")

    print("\nAll conversions attempted.\n")
