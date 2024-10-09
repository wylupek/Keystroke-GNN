import sqlite3


def drop_table() -> bool:
    """
    Drop the key_press table if it exists.

    Returns:
        bool: True if the table was dropped successfully, False otherwise.
    """

    try:
        conn = sqlite3.connect('key_press_data.db')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS key_press')
        conn.commit()
        conn.close()
        print("Table 'key_press' has been dropped (if it existed).")
        return True
    except sqlite3.Error as e:
        conn.close()
        print(f"An error occurred while dropping the table: {e}")
        return False


def setup_database() -> bool:
    """
    Create the key_press_data.db database and the key_press table if they do not exist.

    Returns:
        bool: True if the table was created successfully or already exists, False if there was an error.
    """

    try:
        conn = sqlite3.connect('key_press_data.db')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS key_press (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                press_time TIMESTAMP NOT NULL,
                duration INTEGER NOT NULL,
                date DATE NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database setup complete.")
        return True
    except sqlite3.Error as e:
        conn.close()
        print(f"An error occurred while setting up the database: {e}")
        return False


def add_csv_values(content: list[dict], user_id: str) -> bool:
    """
    Insert records into the key_press table in the key_press_data.db database.

    Args:
        content (list[dict]): A list of dictionaries containing key press data,
                              where each dictionary must contain the keys
                              'key', 'press_time', and 'duration'.
        user_id (str): The ID of the user associated with the key presses.

    Returns:
        bool: True if the records were inserted successfully,
              False if there was an error during the process.
    """

    try:
        conn = sqlite3.connect('key_press_data.db')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return False

    try:
        cursor = conn.cursor()

        # Prepare and execute the insert statements
        for entry in content:
            cursor.execute('''
                INSERT INTO key_press (user_id, key, press_time, duration, date)
                VALUES (?, ?, ?, ?, DATE('now'))
            ''', (
                user_id,
                entry["key"],
                entry["press_time"],
                entry["duration"]
            ))

        conn.commit()
        conn.close()
        print(f"Inserted {len(content)} records successfully.")
        return True
    except sqlite3.Error as e:
        conn.close()
        print(f"An error occurred while adding CSV values: {e}")
        return False


def print_csv(content: list[dict]) -> None:
    print("key\tpress_time\t\tduration\t")
    for row in content:
        print(f"{row['key']}\t{row['press_time']}\t{row['duration']}")
