import csv

def save_summary_to_csv(filename, total, occupied, vacant):
    headers = ['Total number of Slots', 'Occupied Slots', 'Available Slots']
    row = [total, occupied, vacant]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)
