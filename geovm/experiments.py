import csv

rows = []

with open('photo_metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for index, row in enumerate(csv_reader):
        if index == 10:
            break
        else:
            rows.append(row)


with open('test_schema.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(rows[0])
