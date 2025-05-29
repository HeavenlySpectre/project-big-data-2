import csv

def cut_csv_rows(input_filename, output_filename, num_rows_to_keep):
    """
    Cuts a CSV file to a specified number of rows.

    Args:
        input_filename (str): The path to the input CSV file.
        output_filename (str): The path to save the new (cut) CSV file.
        num_rows_to_keep (int): The number of rows to keep from the input file
                                (including the header if present).
    """
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for i, row in enumerate(reader):
                if i < num_rows_to_keep:
                    writer.writerow(row)
                else:
                    break  # Stop reading once we've reached the desired number of rows
        print(f"Successfully created '{output_filename}' with {num_rows_to_keep} rows.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- How to use ---
    input_csv = "recommendations.csv"  # 👈 Replace with your input CSV file name
    output_csv = "realrecommendations.csv" # 👈 Replace with your desired output CSV file name
    rows_to_keep = 5000000                 # 👈 Replace with the number of rows you want

    if rows_to_keep <= 0:
        print("Error: Number of rows to keep must be a positive integer.")
    else:
        cut_csv_rows(input_csv, output_csv, rows_to_keep)