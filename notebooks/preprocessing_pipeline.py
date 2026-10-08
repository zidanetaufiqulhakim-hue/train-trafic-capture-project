import re
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr


class VideoUIPipeline:
    """
    Pipeline for extracting, detecting, and structuring UI tabular data
    directly from video inputs into a pandas DataFrame, CSV, and optional annotated output video.

    station_name and station_numbering are no longer OCR'd or boxed in the
    preview/annotated video. station_name is derived purely from STATION_LOOKUP
    keyed on station_id alone.
    """

    DEFAULT_CLASS_NAMES = [
        'passenger_increse',
        'station_id',
        'station_name',
        'station_numbering',
        'time'
    ]

    # Bounding box color palette (BGR) for distinct visualization per class.
    # 'station_name' and 'station_numbering' intentionally have no entry --
    # both are skipped entirely in OCR and in drawing. station_name is derived
    # from the lookup table; station_numbering is no longer needed at all now
    # that the lookup key is station_id alone.
    CLASS_COLORS = {
        'passenger_increse': (0, 255, 0),    # Green
        'station_id': (255, 0, 0),           # Blue
        'time': (0, 165, 255)                # Orange
    }

    # station_id -> station name mapping (used instead of OCR'ing the name).
    # NOTE: 'BD103' was ambiguous in the original (station_numbering, e.g.
    # '418' vs '417', distinguished 'Pungkur' from 'Asia Afrika'). Keying on
    # station_id alone can't tell these apart anymore -- defaulted to
    # 'Pungkur' here. Fix this entry if that's wrong for your data.
    STATION_LOOKUP = {
        'BD093': 'Dayeuh Kolot',
        'BD095': 'Bojongasih',
        'BD096': 'Pasawahan',
        'BD097': 'Mengger',
        'BD098': 'Wates',
        'BD099': 'Mekar Agung',
        'BD100': 'Moch. Toha',
        'BD101': 'Pasar Montana',
        'BD102': 'Tegal Lega',
        'BD103': 'Pungkur',  # ambiguous -- see note above
        'BD105': 'Banceuy',
        'BD106': 'Bandung',
        'BD107': 'Pasir Kaliki',
        'BD108': 'Hasan Sadikin',
        'BD109': 'Sukabungah',
        'BD110': 'PVJ',
        'BD112': 'Geger Kalong',
        'BD113': 'UPI',
        'BD114': 'Terminal Ledeng',
        'BD115': 'Isola',
        'BD116': 'Isolakahuripan',
        'BD117': 'Gudangkahuripan',
        'BD118': 'Kolonel Masturi',
        'BD119': 'Manoko',
        'BD120': 'Lembang',
    }

    def __init__(
        self,
        yolo_model_path: str = "yolov8s.pt",
        class_names: list = None,
        use_gpu: bool = False,
        ocr_languages: list = ['en'],
        station_id_name_mapping: dict = None,
        station_id_max_edit_distance: int = 2
    ):
        self.yolo_model_path = yolo_model_path
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.use_gpu = use_gpu
        self.ocr_languages = ocr_languages
        self.station_id_max_edit_distance = station_id_max_edit_distance

        self.yolo_model = None
        self.ocr_reader = None

        # Allow overriding/extending the built-in lookup table if needed
        if station_id_name_mapping:
            self.STATION_LOOKUP = {**self.STATION_LOOKUP, **station_id_name_mapping}

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Standard edit distance (insertions/deletions/substitutions) between two strings."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, start=1):
                curr[j] = min(
                    prev[j] + 1,        # deletion
                    curr[j - 1] + 1,    # insertion
                    prev[j - 1] + (ca != cb)  # substitution (0 if match)
                )
            prev = curr
        return prev[-1]

    def _closest_valid_station_id(self, candidate: str):
        """Snaps a noisy OCR'd station id to the nearest id in the known, closed
        set of valid station codes (STATION_LOOKUP keys), using edit distance.
        This is far more robust than pattern-based digit fixes (like truncating
        to N digits) because it doesn't assume *where* the OCR error happened --
        extra digits, missing digits, and substituted digits are all just "edit
        distance" and get handled the same way.

        Returns None (rather than guessing) when nothing is close enough, or
        when two or more valid ids are tied for closest -- an ambiguous read is
        worse than a missing one, since a wrong station id silently corrupts data."""
        if not candidate:
            return None

        best_ids = []
        best_dist = self.station_id_max_edit_distance + 1

        for valid_id in self.STATION_LOOKUP:
            dist = self._levenshtein(candidate, valid_id)
            if dist < best_dist:
                best_dist = dist
                best_ids = [valid_id]
            elif dist == best_dist:
                best_ids.append(valid_id)

        if best_dist > self.station_id_max_edit_distance or len(best_ids) != 1:
            return None
        return best_ids[0]

    def load_models(self):
        """Loads YOLO and EasyOCR models, dynamically fetching class names from model metadata."""
        if self.yolo_model is None:
            self.yolo_model = YOLO(self.yolo_model_path)
            if hasattr(self.yolo_model, 'names') and self.yolo_model.names:
                self.class_names = [self.yolo_model.names[i] for i in range(len(self.yolo_model.names))]

        if self.ocr_reader is None:
            self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=self.use_gpu)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Grayscale, mild upscale, and contrast enhancement without harsh sharpening artifacts."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Upscale 2x for OCR readability
        height, width = gray.shape
        resized = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

        # Contrast adjustment (CLAHE) instead of high-frequency kernel sharpening
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for YOLO compatibility

    def clean_time_text(self, value: str):
        """Parses OCR time text into a normalized HH:MM or HH:MM:SS format."""
        if not value:
            return None

        digits = re.findall(r'\d+', value)

        # Handles HH:MM or MM:SS (2 digit groups)
        if len(digits) == 2:
            val1, val2 = int(digits[0]), int(digits[1])
            return f"{val1:02d}:{val2:02d}"

        # Handles SS HH MM or HH:MM:SS (3 digit groups)
        elif len(digits) >= 3:
            sec, hour, minute = digits[0], digits[1], digits[2]
            try:
                h, m, s = int(hour) % 24, int(minute) % 60, int(sec) % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
            except ValueError:
                return None

        return None

    def clean_station_id_text(self, value: str):
        """Cleans OCR text for a station ID, then snaps it to the nearest id in
        the known, closed set of valid station codes via _closest_valid_station_id.
        Returns None if the OCR text is empty, unparseable, or too far/ambiguous
        to confidently match a real station id -- see _closest_valid_station_id
        for why guessing is avoided rather than forced."""
        if not value:
            return None

        cleaned = re.sub(r'[^A-Za-z0-9]', '', value)

        match = re.match(r'^([A-Za-z]*)(.*)$', cleaned)
        prefix, rest = match.groups()

        rest_corrected = rest.replace('B', '8').replace('O', '0').replace('S', '5')

        digits_match = re.search(r'\d+', rest_corrected)
        if not digits_match:
            return None

        raw_candidate = "BD" + digits_match.group()
        return self._closest_valid_station_id(raw_candidate)

    def passenger_increse_text(self, value: str) -> int:
        """Cleans OCR text for passenger increase, removing non-numeric characters and normalizing."""
        if not value:
            return None
        cleaned = re.sub(r'[^0-9]', '', value)  # remove non-numeric characters
        return int(cleaned) if cleaned else None

    def get_station_name(self, station_id):
        """Looks up station_name purely from station_id."""
        if not station_id:
            return None
        return self.STATION_LOOKUP.get(station_id)

    def process_frame_detections(self, frame: np.ndarray, timestamp: float, row_y_tolerance: int = 20):
        """Detects bounding boxes, extracts OCR text, then groups boxes into
        visual rows by Y-position -- since one frame can contain multiple
        simultaneous passenger entries (each a separate real record), not just one.

        'station_name' and 'station_numbering' boxes are both skipped entirely
        (no OCR, no row entry) -- station_name is derived afterwards from
        station_id alone via the lookup table, and station_numbering is no
        longer needed for anything."""
        if self.yolo_model is None or self.ocr_reader is None:
            self.load_models()

        results = self.yolo_model(frame, conf=0.3, verbose=False)[0]

        SKIPPED_CLASSES = {'station_name', 'station_numbering'}

        detections = []  # (y_center, cls_name, cleaned_text)
        ocr_by_box = {}

        for i, box in enumerate(results.boxes):
            cls_idx = int(box.cls[0])
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"

            # Skip station_name/station_numbering entirely: no crop, no OCR,
            # no row entry, no box drawn later
            if cls_name in SKIPPED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            y_center = (y1 + y2) / 2

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                enhanced_crop = self.preprocess_frame(crop)
                ocr_res = self.ocr_reader.readtext(enhanced_crop, detail=0)
            else:
                ocr_res = []
            ocr_text = " ".join(ocr_res) if ocr_res else ""

            cleaned_text = self.clean_time_text(ocr_text) if cls_name == 'time' else \
                        self.clean_station_id_text(ocr_text) if cls_name == 'station_id' else \
                        self.passenger_increse_text(ocr_text) if cls_name == 'passenger_increse' else \
                        ocr_text.strip()

            ocr_by_box[i] = cleaned_text
            detections.append((y_center, cls_name, cleaned_text))

        # Extract the top HUD time globally for the entire frame
        frame_time = next((text for _, cls_name, text in detections if cls_name == 'time' and text), None)

        # Filter out the 'time' detections from Y-row grouping so the HUD clock doesn't create its own empty row
        table_detections = [d for d in detections if d[1] != 'time']

        # Group table detections into rows by Y-proximity
        table_detections.sort(key=lambda d: d[0])
        rows = []
        current_row = {}
        current_y = None

        for y_center, cls_name, cleaned_text in table_detections:
            if current_y is not None and abs(y_center - current_y) > row_y_tolerance:
                rows.append(current_row)
                current_row = {}
            current_row[cls_name] = cleaned_text
            current_y = y_center

        if current_row:
            rows.append(current_row)

        # Fill each row into the full schema. station_name is no longer read from OCR --
        # it's derived from station_id alone via the lookup table.
        ocr_schema = ['passenger_increse', 'station_id', 'time']
        row_dicts = []
        for row in rows:
            row_dict = {'timestamp': timestamp}
            for key in ocr_schema:
                row_dict[key] = frame_time if key == 'time' else row.get(key)

            row_dict['station_name'] = self.get_station_name(
                row_dict.get('station_id')
            )
            row_dicts.append(row_dict)

        return row_dicts, results.boxes, ocr_by_box

    def draw_detections(self, frame: np.ndarray, raw_boxes, ocr_by_box: dict) -> np.ndarray:
        """Draws bounding boxes, class labels, and OCR values on the frame.
        Reuses OCR results computed in process_frame_detections instead of
        re-running OCR (which previously doubled per-frame latency).
        station_name and station_numbering boxes are skipped -- they were
        never OCR'd, so there's nothing to box or label for either class."""
        annotated_frame = frame.copy()

        for i, box in enumerate(raw_boxes):
            cls_idx = int(box.cls[0])
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"

            if cls_name in ('station_name', 'station_numbering'):
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cleaned_text = ocr_by_box.get(i, None)

            color = self.CLASS_COLORS.get(cls_name, (0, 0, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name}: {cleaned_text} ({conf:.2f})"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(
                annotated_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame

    def temporal_deduplication(self, df: pd.DataFrame, max_gap_seconds: float = 5.0) -> pd.DataFrame:
        """Removes consecutive duplicate rows, handling NaNs and matching on key primary identifiers."""
        if df.empty:
            return df

        # Deduplicate primarily using robust identifiers (ignore 'time' since it's a HUD clock, not a key)
        key_cols = ['station_id', 'passenger_increse']
        cols_present = [c for c in key_cols if c in df.columns]

        if not cols_present:
            return df

        # Fill NaNs temporarily with a dummy string so NaN == NaN evaluates as True
        filled_df = df[cols_present].fillna("__MISSING__")

        # Check string equality against previous row
        is_same_as_prev = (filled_df == filled_df.shift(1)).all(axis=1)
        time_diff = df['timestamp'] - df['timestamp'].shift(1)

        is_duplicate = is_same_as_prev & (time_diff <= max_gap_seconds)
        return df[~is_duplicate].reset_index(drop=True)

    def drop_same_timestamp_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """A single video frame's row-grouping sometimes yields more than one row
        for the exact same timestamp -- typically one full/real row plus a partial
        echo (e.g. a misread or half-rendered badge like 'BD15' sitting next to a
        real 'BD108/413' row). Keep only the most complete row per timestamp and
        drop the rest.

        Note: if your on-screen UI can legitimately show two distinct real entries
        within one frame, this will still collapse them to one -- there's no way to
        tell a genuine second entry from a duplicate echo once both are complete."""
        if df.empty or 'timestamp' not in df.columns:
            return df

        key_cols = [c for c in ['passenger_increse', 'station_id', 'station_name']
                    if c in df.columns]
        if not key_cols:
            return df

        completeness = df[key_cols].notna().sum(axis=1)
        scored = df.assign(_completeness=completeness)

        # Stable sort: within a timestamp, highest completeness first; ties keep
        # original (first-seen) order. Timestamps stay in their original relative order.
        scored = scored.sort_values(['timestamp', '_completeness'], ascending=[True, False], kind='mergesort')

        deduped = scored.drop_duplicates(subset='timestamp', keep='first').drop(columns='_completeness')
        return deduped.reset_index(drop=True)

    def drop_missing_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops rows where all key OCR'd columns are missing. station_name is excluded
        here since it's a derived field, not an OCR result -- it shouldn't decide whether
        a row gets dropped."""
        key_cols = ['passenger_increse', 'station_id']
        existing_cols = [c for c in key_cols if c in df.columns]

        return df.dropna(subset=existing_cols, how='all').reset_index(drop=True)

    def impute_missing_station_id_and_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills "Null" in both columns as "OCR MISSREAD" for station_name based on station_id. This is a post-processing step"""
        if 'station_id' in df.columns and 'station_name' in df.columns:
            df['station_name'] = df.apply(
            lambda row: self.get_station_name(row['station_id']) if pd.notna(row['station_id']) else "UNKNOWN ID",
                axis=1
            )

            df['station_id'] = df['station_id'].fillna("OCR MISSREAD")

        return df

    def run(
        self,
        video_path: str,
        target_fps: float = 2.0,
        export_csv: bool = True,
        output_csv_path: str = "output.csv",
        show_preview: bool = True,
        save_annotated_video: bool = False,
        output_video_path: str = "annotated_output.mp4"
    ) -> pd.DataFrame:
        """Reads video, processes detections, cleans rows, and deduplicates cleanly."""
        self.load_models()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        video_writer = None
        frame_interval = int(max(1, video_fps / target_fps))
        all_rows = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / video_fps
                rows, raw_boxes, ocr_by_box = self.process_frame_detections(frame, timestamp_sec)
                all_rows.extend(rows)

                annotated_frame = self.draw_detections(frame, raw_boxes, ocr_by_box)

                if save_annotated_video:
                    if video_writer is None:
                        h, w = annotated_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, (w, h))
                    video_writer.write(annotated_frame)

                if show_preview:
                    cv2.imshow("UI Object Detection Preview", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_idx += 1

        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Final Post-Processing Pipeline
        raw_df = pd.DataFrame(all_rows)

        # 1. Collapse multiple rows from the same frame/timestamp into the single
        #    most complete row (drops partial echoes like the 'BD15' fragment case)
        collapsed_df = self.drop_same_timestamp_duplicates(raw_df)

        # 2. Drop rows where all key OCR fields were completely undetected
        clean_df = self.drop_missing_rows(collapsed_df)

        # 3. Deduplicate consecutive identical frame outputs within target timeframe
        final_df = self.temporal_deduplication(clean_df, max_gap_seconds=5.0)

        # 4. Impute station_name from station_id, marking missing station_id as "OCR MISSREAD"
        final_df = self.impute_missing_station_id_and_names(final_df)

        if export_csv:
            final_df.to_csv(output_csv_path, index=False)

        return final_df


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    pipeline = VideoUIPipeline(
        yolo_model_path="notebooks/runs/detect/train_passenger_capture/yolov8s_mps_run/weights/best.pt",
        use_gpu=False
    )

    result_df = pipeline.run(
        video_path="notebooks/sample_video.mp4",
        target_fps=1.0,
        show_preview=True,           # Opens a live window showing bounding boxes
        save_annotated_video=True,   # Saves the detection result to annotated_output.mp4
        output_video_path="detection_preview.mp4"
    )

    print("\nProcessed Output Preview:")
    print(result_df.head())