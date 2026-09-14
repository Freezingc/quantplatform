"""Fetch the FULL PIT membership table from index_member_all (both is_new halves).

Read-only against the vendor. Writes only under --out-dir. No database, no repo file touched.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

ENV = Path('/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/.env')
FIELDS = 'l1_code,l1_name,ts_code,in_date,out_date,is_new,src'


def token() -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith('QRV2_TUSHARE_TOKEN='):
            return line.split('=', 1)[1].strip()
    raise SystemExit('no QRV2_TUSHARE_TOKEN')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--codes', nargs='*', default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import tushare as ts
    pro = ts.pro_api(token())

    cls_path = args.out_dir / 'sw2021_l1_classification.json'
    if cls_path.is_file():
        cls = json.loads(cls_path.read_text())
    else:
        dfc = pro.index_classify(src='SW2021', level='L1',
                                 fields='index_code,industry_name,level,parent_code,src')
        cls = dfc.to_dict(orient='records')
        cls_path.write_text(json.dumps(cls, ensure_ascii=False, indent=1))
    codes = args.codes or [r['index_code'] for r in cls]
    print('L1 codes:', len(codes))

    rows: list[dict] = []
    for code in codes:
        for is_new in ('Y', 'N'):
            cache = args.out_dir / f"member_all_{code.replace('.', '_')}_{is_new}.json"
            if cache.is_file():
                part = json.loads(cache.read_text())
            else:
                for attempt in range(4):
                    try:
                        df = pro.index_member_all(l1_code=code, is_new=is_new, fields=FIELDS)
                        break
                    except Exception as exc:  # rate limit / transient
                        print('  retry', code, is_new, repr(exc)[:120]); time.sleep(3 * (attempt + 1))
                else:
                    raise SystemExit(f'failed {code} {is_new}')
                part = df.to_dict(orient='records')
                cache.write_text(json.dumps(part, ensure_ascii=False))
                time.sleep(0.35)
            rows.extend(part)
            print(f'{code} is_new={is_new}: {len(part)}')
    (args.out_dir / 'member_all_combined.json').write_text(json.dumps(rows, ensure_ascii=False))
    print('TOTAL rows', len(rows))


if __name__ == '__main__':
    main()
