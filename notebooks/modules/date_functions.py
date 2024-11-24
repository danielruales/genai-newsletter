from datetime import datetime

def get_current_date():
    # Get the current date
    current_date = datetime.now()
    # Format the date as YYYY-MM-DD
    formatted_date = current_date.strftime('%Y-%m-%d')
    return formatted_date