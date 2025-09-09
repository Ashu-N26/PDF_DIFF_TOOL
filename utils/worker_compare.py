# utils/worker_compare.py
import os
import sys
import json
import traceback
from utils.pdf_diff import compare_pdfs

def main():
    if len(sys.argv) < 5:
        print("usage: worker_compare.py old.pdf new.pdf out_dir prefix", file=sys.stderr)
        sys.exit(2)
    old_path, new_path, out_dir, prefix = sys.argv[1:5]
    try:
        # call compare with a conservative scale default; pdf_diff accepts scale parameter if implemented
        outputs = compare_pdfs(old_path, new_path, out_dir, prefix=prefix)
        # outputs already map to filenames; write outputs.json
        out_json = os.path.join(out_dir, f"{prefix}outputs.json")
        outputs_with_paths = {}
        for k,v in outputs.items():
            outputs_with_paths[k] = v
        outputs_with_paths['status'] = 'ok'
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(outputs_with_paths, fh)
        print("done")
        sys.exit(0)
    except Exception as e:
        err = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        try:
            out_json = os.path.join(out_dir, f"{prefix}outputs.json")
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(err, fh)
        except Exception:
            pass
        print("error", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
