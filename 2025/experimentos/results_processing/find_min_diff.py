import csv
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_min_diff.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    best_row = None
    best_diff = float('inf')

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frr = float(row['frr'])
            far = float(row['far'])
            diff = abs(frr - far)
            if diff < best_diff:
                best_diff = diff
                best_row = row

    if best_row is None:
        print("No data found")
        sys.exit(1)

    frr = float(best_row['frr'])
    far = float(best_row['far'])
    avg = (frr + far) / 2.0
    threshold = best_row['threshold']

    print(f"Row with minimal |frr - far| (diff = {best_diff:.10f}):")
    print(f"  frr = {frr}")
    print(f"  far = {far}")
    print(f"  Average = {avg}")
    print(f"  Threshold = {threshold}")

if __name__ == '__main__':
    main()
