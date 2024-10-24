import os
import json
from groq import Groq
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
import time
# Ensure the punkt tokenizer is available
nltk.download('punkt')
os.environ['GROQ_API_KEY'] = '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'


# Step 1: Extract common y-coordinates (for headers/footers) and text blocks
def extract_common_y_coords(pdf_path):
    doc = fitz.open(pdf_path)

    # Check if the PDF has more than one page
    if len(doc) <= 1:
        print("PDF contains only one page. No processing needed.")
        return None, doc

    y_coords_all_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        y_coords = [block[1] for block in blocks]  # Get y-coordinates
        y_coords_all_pages.append(set(y_coords))   # Store as set for easier intersection

    # Find common y-coordinates across all pages
    common_y_coords = set.intersection(*y_coords_all_pages)

    return common_y_coords, doc

# Step 2: Remove text from pages based on common y-coordinates
def remove_header_footer(doc, common_y_coords, tolerance=10):
    if common_y_coords is None:
        print("Skipping redaction since no common y-coordinates were found.")
    else:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")

            for block in blocks:
                block_y_coord = block[1]
                # Check if the block's y-coordinate matches the common y-coordinates (allowing small tolerance)
                if any(abs(block_y_coord - common_y) < tolerance for common_y in common_y_coords):
                    # Mark the block for redaction (removal)
                    page.add_redact_annot(block[:4], fill=(1, 1, 1))  # Redact with white fill
            # Apply redactions for the current page
            page.apply_redactions()

    return doc

# Function to calculate the centroid of a text block
def getCentroid(bbox):
    y0, y1 = bbox[1], bbox[3]
    return (y0 + y1) / 2  # Centroid is the middle of y0 and y1

# Adjust the centroids of text spans based on adjacent text
def adjustCentroid(centroids):
    adjustedCentroids = centroids.copy()
    for i in range(1, len(centroids)):
        prevCentroid = centroids[i - 1]['centroid']
        currCentroid = centroids[i]['centroid']
        currY0, currY1 = centroids[i]['y0'], centroids[i]['y1']

        # If the current centroid falls between y0 and y1 of the previous text, adjust it
        if currY0 < prevCentroid < currY1:
            adjustedCentroids[i]['centroid'] = prevCentroid

    return adjustedCentroids

def extract_text(doc):

    # Extract all text from the PDF and organize it by page in the order as it appears
    pdfTextData = {}
    for pageNum in range(doc.page_count):
        page = doc.load_page(pageNum)
        blocks = page.get_text("dict")["blocks"]

        pageTextData = []
        for block in blocks:
            if block.get("type") == 0:  # Only process text blocks (not images, lines, etc.)
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        bbox = span["bbox"]
                        xCoord = bbox[0]  # X-coordinate of the text block
                        yCoord = bbox[1]  # Top Y-coordinate
                        centroid = getCentroid(bbox)  # Calculate centroid

                        # if not isInHeaderFooter(text, centroid):
                        pageTextData.append({
                            "text": text,
                            "xCoord": xCoord,
                            "yCoord": yCoord,
                            "y0": bbox[1],  # Top Y-coordinate of the span
                            "y1": bbox[3],  # Bottom Y-coordinate of the span
                            "centroid": centroid
                        })

        # Adjust centroids based on adjacent text spans
        pageTextData = adjustCentroid(pageTextData)

        # Sort the text on the page by the adjusted centroid and X-coordinate
        pageTextData = sorted(pageTextData, key=lambda x: (x['centroid'], x['xCoord']))

        # Join the sorted text into a single string for the page
        sorted_text = " ".join(item['text'] for item in pageTextData)
        pdfTextData[pageNum + 1] = sorted_text  # Store sorted text for each page


        # pdfTextData[pageNum + 1] = pageTextData  # Store text for each page in order

    # Create a DataFrame to save the extracted text
    outputData = {
        "Page Number": [],
        "Text": []
    }

    for pageNum, text in pdfTextData.items():
        outputData["Page Number"].append(pageNum)
        outputData["Text"].append(text)

    # Convert to DataFrame and save to CSV
    # outputCsvPath = '/content/extracted_text_by_page.csv'
    outputDf = pd.DataFrame(outputData)
    return outputDf

# Function to normalize sentences (remove leading/trailing spaces and normalize whitespace)
def normalize_sentence(sentence):
    return re.sub(r'\s+', ' ', sentence.strip())


# Function to find the index of a sentence in the full text
def find_index(sentence, full_text):
    try:
        return full_text.index(sentence)
    except ValueError:
        return -1  # Return -1 if the sentence is not found



# Initialize the Groq client
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

# Function to find added, deleted text, and explanation using Groq
def find_added_deleted_with_groq(old_text, new_text):

    # Convert to strings, handle NaN by converting to empty strings
    old_text = str(old_text) if not pd.isna(old_text) else ''
    new_text = str(new_text) if not pd.isna(new_text) else ''

    # Adjusted prompt for clearer response format
    prompt = (
        f"Given the following texts:\n"
        f"Old Text: '{old_text}'\n"
        f"New Text: '{new_text}'\n\n"
        f"Please identify the added and deleted text along with the impact of the changes in meaning on a scale of 1 to 10 where 1 being no change and 10 being major change in meaning in strictly the following JSON format:\n"
        f"{{\n"
        f"  'json_start': 'JSON Starts from here',\n"
        f"  'added_text': '...',\n"
        f"  'deleted_text': '...',\n"
        f"  'Change_summary': '...',\n"
        f"  'Impact': '...',\n"
        f"  'json_end': 'JSON Ends here'\n"
        f"}}"
    )


    retry_attempts = 3
    while retry_attempts > 0:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:  # Catch general exceptions
            print(f"An error occurred: {e}")
            if 'rate_limit_exceeded' in str(e):
                # Try to extract retry time from the message
                retry_match = re.search(r'(\d+m\d+\.\d+s)', str(e))
                if retry_match:
                    retry_time = retry_match.group(1)
                    # Convert to seconds
                    minutes, seconds = map(float, re.findall(r'\d+\.\d+|\d+', retry_time))
                    retry_after = minutes * 60 + seconds
                else:
                    retry_after = 60  # Default to 60 seconds if we can't extract the retry time

                print(f"Rate limit reached. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                retry_attempts -= 1
            else:
                raise e  # If it's another error, raise it

    raise Exception("Exceeded retry limit. Please try again later.")

    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         }
    #     ],
    #     model="llama3-8b-8192",
    # )

    # Extracting added, deleted text, and explanation from the model's response
    response_content = chat_completion.choices[0].message.content
    return response_content

def parse_response(response):

    return '', '', '','', response


# Function to extract only numbers from a string
def extract_numbers(text):
    # Use regex to find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    # Join the numbers into a single string, separated by spaces
    return ' '.join(numbers)

def clean_text(text):
    if isinstance(text, str):
        # Remove any symbols before the first alphanumeric character and after the last alphanumeric character
        cleaned = re.sub(r'^[^\w]+', '', text)  # Remove leading non-alphanumeric characters
        cleaned = re.sub(r'[^\w]+$', '', cleaned)  # Remove trailing non-alphanumeric characters
        return cleaned
    else:
        return text  # If it's not a string, return the original value


# Function to extract text between two words
def extract_text_between(text, start_word, end_word):
    try:
        start_index = text.index(start_word) + len(start_word)
        end_index = text.index(end_word, start_index)
        return text[start_index:end_index].strip()
    except ValueError:
        return None  # Return None if the words are not found

# Function to split text into coherent sentences
def split_text_into_coherent_sentences(text):
    # Find all sequences of words and numbers that are separated by spaces or punctuation
    sentences = []
    current_sentence = []

    # Ensure the text is a string
    if not isinstance(text, str):
        return []

    # Split the text into words
    words = re.findall(r'\S+', text)  # This finds sequences of non-whitespace characters

    for word in words:
        # Check for specific patterns to identify sentence boundaries
        if word in ['.', ';', ':', ',']:  # If the word is punctuation, continue
            continue
        elif re.match(r'^\d', word):  # If it starts with a digit, we can consider it as part of the sentence
            current_sentence.append(word)
        elif word.startswith('('):  # Handle parenthesis
            continue
        else:
            current_sentence.append(word)

        # Add the current sentence if a new sentence should start
        if len(current_sentence) > 0 and (len(current_sentence) >= 5 or word.endswith('.')):  # Arbitrary sentence end logic
            sentences.append(' '.join(current_sentence).strip())
            current_sentence = []  # Reset for the next sentence

    # Add any remaining words as a final sentence
    if current_sentence:
        sentences.append(' '.join(current_sentence).strip())

    return sentences


# Function to find the y-coordinate of each sentence in the PDF
def find_sentence_locations(pdf_document, sentence):
    locations = []
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text_instances = page.search_for(sentence)
        for inst in text_instances:
            # inst is a rectangle, we want the y-coordinate
            y_coordinate = inst[1]  # the y-coordinate of the top left corner
            locations.append((page_number + 1, y_coordinate))  # page number is 1-indexed
    return locations


# Define a function to replace 'Page Number' with the most common 'Page Number'
def replace_with_most_common_page_number(group):
    # Find the most common 'Page Number'
    most_common_page = group['Page Number'].mode()[0]
    # Replace the 'Page Number' in the group with the most common one
    group['Page Number'] = most_common_page
    return group

def highlight_pdf(original_pdf_file_path, modified_pdf_file_path, df, text_to_highlight, highlight_color):

    # Load the CSV file

    # Create a list to hold the split sentences
    split_sentences_data = []

    # Iterate through the 'Added Text' column, split the text, and save the sentences
    for index, row in df.iterrows():
        added_text = row.get(text_to_highlight, '')  # Use .get() to safely retrieve the value
        sentences = split_text_into_coherent_sentences(added_text)

        # Append each sentence to the list
        for sentence in sentences:
            split_sentences_data.append({'Original Row': index, 'Split Sentences': sentence})

    # Create a new DataFrame from the list
    split_sentences_df = pd.DataFrame(split_sentences_data)

    df = split_sentences_df.reset_index(drop=True)

    # Load the PDF file
    pdf_file_path = original_pdf_file_path
    pdf_document = fitz.open(pdf_file_path)


    # Store the results
    results = []

    # Iterate through the dataframe
    for _, row in df.iterrows():
        original_row = row['Original Row']
        split_sentence = row['Split Sentences']

        # Get locations for the current split sentence
        locations = find_sentence_locations(pdf_document, split_sentence)

        if locations:
            # Find the closest page for the original row
            page_numbers = [loc[0] for loc in locations]
            most_common_page = max(set(page_numbers), key=page_numbers.count)

            # Filter locations by the most common page
            closest_locations = [loc for loc in locations if loc[0] == most_common_page]

            if closest_locations:
                # Choose the y-coordinate that is closest to 0 (the top of the page)
                chosen_y_coordinate = min(closest_locations, key=lambda x: x[1])[1]
                results.append((original_row, split_sentence, most_common_page, chosen_y_coordinate))

    # Create a DataFrame for results
    results_df = pd.DataFrame(results, columns=['Original Row', 'Split Sentence', 'Page Number', 'Y Coordinate'])



    df = results_df.reset_index(drop=True)

    # Group by 'Original Row' and apply the function
    df_updated = df.groupby('Original Row').apply(replace_with_most_common_page_number)

    # Reset the index (optional)
    df_updated.reset_index(drop=True, inplace=True)


    df = df_updated

    # Load the PDF file
    pdf_file_path = original_pdf_file_path
    pdf_document = fitz.open(pdf_file_path)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        split_sentence = row['Split Sentence']
        page_number = int(row['Page Number']) - 1  # Adjust for zero-based indexing
        y_coordinate = row['Y Coordinate']

        # Get the specific page
        page = pdf_document[page_number]

        # Search for the split sentence in the page
        found = False
        text_instances = page.search_for(split_sentence)

        for rect in text_instances:
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=highlight_color)
            highlight.update()
            # print(f"Highlighted text at {rect}.")



        # # If the sentence is found, highlight it
        # for rect in text_instances:
        #     # Highlight text in the same Y coordinate
        #     # text_y_coordinate = rect.y0  # y0 is the bottom y-coordinate of the rectangle
        #     # for text_block in page.get_text("blocks"):
        #         # if abs(text_block[1] - text_y_coordinate) < 1e-5:  # y0 of the block
        #             highlight = page.add_highlight_annot(fitz.Rect(text_block[:4]))  # Highlight the whole block
        #             highlight.set_colors(stroke=highlight_color)  # Set highlight color to green
        #             highlight.update()


    # Save the modified PDF
    output_pdf_path = modified_pdf_file_path
    pdf_document.save(output_pdf_path)
    # pdf_document.close()

    print(f"Highlighted PDF saved as {output_pdf_path}.")



def main(pdf_file_path_old, pdf_file_path_new, modified_output_pdf_file_path_old, modified_output_pdf_file_path_new):

    # Step 1: Extract common y-coordinates (headers/footers)
    common_y_coords, new_doc = extract_common_y_coords(pdf_file_path_new)

    # Step 2: Remove the text at those y-coordinates
    new_doc = remove_header_footer(new_doc, common_y_coords)

    # Step 1: Extract common y-coordinates (headers/footers)
    common_y_coords, old_doc = extract_common_y_coords(pdf_file_path_old)

    # Step 2: Remove the text at those y-coordinates
    old_doc = remove_header_footer(old_doc, common_y_coords)

    new_df = extract_text(new_doc)
    original_new_df = new_df.copy()
    old_df = extract_text(old_doc)
    original_old_df = old_df.copy()

    # Concatenate all text into a single string
    text_new = ' '.join(new_df['Text'])
    text_old = ' '.join(old_df['Text'])

    # Tokenize the texts into sentences
    sentences_new = sent_tokenize(text_new)
    sentences_old = sent_tokenize(text_old)

    # Create lists to hold common sentences from each source
    common_sentences_new = []
    common_sentences_old = []

    # Normalize all sentences
    normalized_sentences_new = [normalize_sentence(sentence) for sentence in sentences_new]
    normalized_sentences_old = [normalize_sentence(sentence) for sentence in sentences_old]

    # Find common sentences using regex
    for sentence in normalized_sentences_new:
      for compare_sentence in normalized_sentences_old:
          # Use regex to find a match (ignoring case and allowing for slight variations)
          pattern = re.escape(sentence)  # Escape the sentence to safely use in regex
          if re.search(pattern, compare_sentence, re.IGNORECASE):
              common_sentences_new.append(sentences_new[normalized_sentences_new.index(sentence)])  # Original sentence
              common_sentences_old.append(sentences_old[normalized_sentences_old.index(compare_sentence)])  # Original sentence
              break  # Break once a match is found to avoid duplicates

    # Create a DataFrame for the common sentences
    # The length of the two lists might differ, so we'll pad the shorter list with NaN
    max_length = max(len(common_sentences_new), len(common_sentences_old))

    # Create a DataFrame with separate columns for each source
    common_df = pd.DataFrame({
      'Common Sentence in text_new': common_sentences_new + [None] * (max_length - len(common_sentences_new)),
      'Common Sentence in text_old': common_sentences_old + [None] * (max_length - len(common_sentences_old))
    })


    # Drop rows where the sentences in both columns do not match exactly
    filtered_df = common_df[common_df['Common Sentence in text_new'] == common_df['Common Sentence in text_old']]

    # Optionally, reset the index of the filtered DataFrame
    filtered_df.reset_index(drop=True, inplace=True)

    # Concatenate all text into a single string
    text_new = ' '.join(new_df['Text'])
    text_old = ' '.join(old_df['Text'])

    # Find and save the index positions
    filtered_df['index_new'] = filtered_df['Common Sentence in text_new'].apply(lambda sentence: find_index(sentence, text_new))
    filtered_df['index_old'] = filtered_df['Common Sentence in text_old'].apply(lambda sentence: find_index(sentence, text_old))

    #######################################

    df = filtered_df.reset_index(drop=True)
    # Loop to ensure both 'index_new' and 'index_old' values are strictly increasing
    while True:
      # Identify rows where the conditions are not satisfied
      condition_new = df['index_new'].shift(-1) <= df['index_new']
      condition_old = df['index_old'].shift(-1) <= df['index_old']

      # Combine conditions to find rows to drop
      to_drop = df[condition_new | condition_old]

      # # Print the values of the indexes being dropped
      # if not to_drop.empty:
      #     print("Dropping the following rows:")
      #     print(to_drop[['index_new', 'index_old']])

      # Drop rows that do not satisfy the strictly increasing condition for both indexes
      df = df[~(condition_new | condition_old)]

      # Break the loop if no rows were dropped in this iteration
      if to_drop.empty:
          break
    # filtered_common_sentences_in_order_df = df
    filtered_common_sentences_in_order_df = df.reset_index(drop=True)
    # # Save the updated DataFrame to a new CSV file
    # df.to_csv('/content/filtered_common_sentences_in_order.csv', index=False)

    ##########################


    # # Load the common sentences file for new text
    common_sentences_new_df = filtered_common_sentences_in_order_df

    # # Load the extracted text by page file for new text
    extracted_text_new_df = original_new_df



    # Concatenate all text into one large text for new text
    text_new = ' '.join(extracted_text_new_df['Text'].tolist())

    # Prepare to store extracted sentences for new text
    extracted_sentences_new = []

    # Track the last found index for new sentences
    last_index_new = 0

    # Iterate through each common sentence for new text
    for i in range(len(common_sentences_new_df)):
        sentence_1 = str(common_sentences_new_df['Common Sentence in text_new'][i]) if pd.notna(common_sentences_new_df['Common Sentence in text_new'][i]) else ""

        if i < len(common_sentences_new_df) - 1:  # Only look for adjacent pairs if not at the last sentence
            sentence_2 = str(common_sentences_new_df['Common Sentence in text_new'][i + 1]) if pd.notna(common_sentences_new_df['Common Sentence in text_new'][i + 1]) else ""

            # Create regex pattern
            pattern = re.escape(sentence_1) + r'(.*?)' + re.escape(sentence_2)

            # Search for the pattern in text_new starting from the last found index
            match = re.search(pattern, text_new[last_index_new:], re.DOTALL)

            if match:
                extracted_text = match.group(0)  # Include both sentences
                extracted_sentences_new.append(extracted_text)

                # Update last_index_new to the starting index of the current common sentence
                last_index_new += text_new[last_index_new:].find(sentence_1)
            else:
                extracted_sentences_new.append(None)  # If not found, append None
        else:
            extracted_sentences_new.append(None)  # Append None for the last sentence since it has no adjacent pair

    # Add the extracted sentences for new text to the dataframe
    common_sentences_new_df['extracted sentences new'] = extracted_sentences_new



    # Load the common sentences file for old text
    common_sentences_old_df = filtered_common_sentences_in_order_df

    # Load the extracted text by page file for old text
    extracted_text_old_df = original_old_df


    # Concatenate all text into one large text for old text
    text_old = ' '.join(extracted_text_old_df['Text'].tolist())

    # Prepare to store extracted sentences for old text
    extracted_sentences_old = []

    # Track the last found index for old sentences
    last_index_old = 0

    # Iterate through each common sentence for old text
    for i in range(len(common_sentences_old_df)):
        sentence_1 = str(common_sentences_old_df['Common Sentence in text_old'][i]) if pd.notna(common_sentences_old_df['Common Sentence in text_old'][i]) else ""

        if i < len(common_sentences_old_df) - 1:  # Only look for adjacent pairs if not at the last sentence
            sentence_2 = str(common_sentences_old_df['Common Sentence in text_old'][i + 1]) if pd.notna(common_sentences_old_df['Common Sentence in text_old'][i + 1]) else ""

            # Create regex pattern
            pattern = re.escape(sentence_1) + r'(.*?)' + re.escape(sentence_2)

            # Search for the pattern in text_old starting from the last found index
            match = re.search(pattern, text_old[last_index_old:], re.DOTALL)

            if match:
                extracted_text = match.group(0)  # Include both sentences
                extracted_sentences_old.append(extracted_text)

                # Update last_index_old to the starting index of the current common sentence
                last_index_old += text_old[last_index_old:].find(sentence_1)
            else:
                extracted_sentences_old.append(None)  # If not found, append None
        else:
            extracted_sentences_old.append(None)  # Append None for the last sentence since it has no adjacent pair

    # Add the extracted sentences for old text to the dataframe
    common_sentences_old_df['extracted sentences old'] = extracted_sentences_old

    # Combine the two DataFrames, retaining only relevant columns
    combined_df = pd.DataFrame({
        'Common Sentence in text_new': common_sentences_new_df['Common Sentence in text_new'],
        'extracted sentences new': common_sentences_new_df['extracted sentences new'],
        'Common Sentence in text_old': common_sentences_old_df['Common Sentence in text_old'],
        'extracted sentences old': common_sentences_old_df['extracted sentences old']
    })


    df = combined_df.reset_index(drop=True)

    # Drop rows where either 'extracted sentences new' or 'extracted sentences old' is NaN or empty
    df_cleaned = df.dropna(subset=['extracted sentences new', 'extracted sentences old'])
    df_cleaned = df_cleaned[(df_cleaned['extracted sentences new'] != '') & (df_cleaned['extracted sentences old'] != '')]

    # Find the rows that will be dropped because both 'extracted sentences new' and 'extracted sentences old' are the same
    rows_dropped = df_cleaned[df_cleaned['extracted sentences new'] == df_cleaned['extracted sentences old']]


    # Filter out the rows where both columns are exactly the same
    df_filtered = df_cleaned[df_cleaned['extracted sentences new'] != df_cleaned['extracted sentences old']]

    filtered_extracted_sentences_combined_df = df_filtered.reset_index(drop=True)



    df = filtered_extracted_sentences_combined_df


    if df.empty:
        print("The DataFrame is empty. Filling with default values.")

        # # Assuming 'text_new' and 'text_old' are defined variables
        # text_new = "Your text for text_new here"  # Replace with actual text
        # text_old = "Your text for text_old here"  # Replace with actual text

        # Create a new DataFrame with the specified columns and fill with the required text
        df = pd.DataFrame({

            'extracted sentences new': [text_new],

            'extracted sentences old': [text_old]
        })
    else:
        print("The DataFrame is not empty.")

    # Save the DataFrame to a CSV file
    # df.to_csv('/content/filtered_extracted_sentences_combined.csv', index=False)




    # filtered_extracted_sentences_combined_df.to_csv('/content/filtered_extracted_sentences_combined.csv', index=False)

    # Apply the function to each row of the DataFrame
    added_deleted_results = df.apply(
      lambda row: find_added_deleted_with_groq(row['extracted sentences old'], row['extracted sentences new']),
      # lambda row: find_added_deleted_with_groq(row['Old Start Heading'], row['New Start Heading']),
      axis=1
    )

    # Create new columns in the DataFrame
    df['Added Text'], df['Deleted Text'],  df['Change_summary'], df['Impact'],df['JSON Response'] = zip(*added_deleted_results.apply(parse_response))


    # Extract texts and create new columns
    df['Added Text'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'added_text', 'deleted_text'))
    df['Deleted Text'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'deleted_text', 'Change_summary'))
    df['Change_summary'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'Change_summary', 'Impact'))
    df['Impact'] = df['JSON Response'].apply(lambda x: extract_text_between(x, 'Impact', 'json_end'))




    # Apply the function to the 'Impact' column
    df['Impact'] = df['Impact'].apply(lambda x: extract_numbers(str(x)))

    # Drop rows where 'Impact' contains '1' or '1.0' (including variations)
    df_filtered = df[~df['Impact'].astype(str).str.contains(r'^\s*(1|1\.0)\s*(\(.*\))?$', na=False, case=False)]

    df = df_filtered.reset_index(drop=True)
    
    # Apply the cleaning function to the specified columns
    columns_to_clean = ['Added Text', 'Deleted Text', 'Change_summary']
    for column in columns_to_clean:
      df[column] = df[column].apply(clean_text)

    #################### generating output csv file with Added, Deleted, Summary and Impact Score
    df.dropna(inplace=True)
    summary = df['Change_summary'].tolist()
    try:
        impact_score_list = df['Impact'].tolist()
        print(impact_score_list)
        max_score = max(impact_score_list)
        max_score = int(max_score)
        if max_score<4:
            impact_level='Low'
        elif max_score<7:
            impact_level = 'Medium'
        elif max_score<=10:
            impact_level='High' 
    except:
        impact_level = 'Low'

    

    # df.to_csv('/content/cleaned_filtered_differences.csv', index=False)


    ###Highlighting and generating PDF file - New
    highlight_color = (0, 1, 0)

    highlight_pdf(pdf_file_path_new , modified_output_pdf_file_path_new, df ,'Added Text',highlight_color)

    ###Highlighting and generating PDF file - Old
    highlight_color = (1, 0, 0)

    highlight_pdf(pdf_file_path_old , modified_output_pdf_file_path_old, df ,'Deleted Text',highlight_color)

    return modified_output_pdf_file_path_new, modified_output_pdf_file_path_old, summary, impact_level

##input files
pdf_file_path_new = r"c:\Users\TVPC0032\Anandhu H\HyperApps\Regulatory Compliance Assistant\Notification_Sample\Chapter4\C4_P407_change.pdf"
pdf_file_path_old = r"c:\Users\TVPC0032\Anandhu H\HyperApps\Regulatory Compliance Assistant\Regulations\Chapter4\C4_P407.pdf"
modified_output_pdf_file_path_new = "new_highlighted.pdf"
modified_output_pdf_file_path_old = "old_highlighted.pdf"
#Output file paths


highlighted_new, highlighted_old, summary, impact_level = main(pdf_file_path_new,pdf_file_path_old, modified_output_pdf_file_path_new, modified_output_pdf_file_path_old)

print(highlighted_new)
print(highlighted_old)
print(summary)
print(impact_level)

print("complete")
