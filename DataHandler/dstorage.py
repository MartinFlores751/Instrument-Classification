import csv


def saveToCSV(filename, features):
    """Saves a 2D array(features) as a CSV file"""
    with open('Extracted_Features/' + filename + 'features.csv', 'w+') as my_csv:
        csv_writer = csv.writer(my_csv)
        csv_writer.writerows(features)

