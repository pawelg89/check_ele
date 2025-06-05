#!/usr/bin/env python3

import requests
import time
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from collections import deque
from datetime import datetime
import threading
import argparse
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('url_check.log'),
        logging.StreamHandler()
    ]
)

CACHE_FILE = 'cache.txt'
RESULTS_FILE = 'results.json'
# Number of recent requests to consider for rate calculation
RATE_WINDOW = 10
# Rate thresholds for detecting slowdown and recovery
RATE_THRESHOLD_LOW = 0.1  # Hz
RATE_THRESHOLD_HIGH = 5.0  # Hz


def setup_driver():
    """Set up and return a headless Chrome WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=chrome_options)


def get_page_content(url, driver):
    """Get the actual page content using Selenium."""
    try:
        driver.get(url)
        # Wait for the content to load (adjust selector based on actual page)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "obkw"))
        )
        # Give a little extra time for all content to load
        time.sleep(2)
        content = driver.page_source

        # Save raw HTML content to file
        url_number = url.split('/')[-1]
        html_file = f'pages/page_{url_number}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Saved raw HTML content to {html_file}")

        return content
    except TimeoutException:
        logging.error(f"Timeout waiting for content to load: {url}")
        return None
    except Exception as e:
        logging.error(f"Error loading page {url}: {str(e)}")
        return None


class RateTracker:
    def __init__(self, window_size=10):
        self.timestamps = deque(maxlen=window_size)
        # Keep last 100 rate measurements
        self.rate_history = deque(maxlen=100)
        self.slowdown_start = None
        self.recovery_times = []

    def add_request(self):
        self.timestamps.append(time.time())
        current_rate = self.get_rate()
        self.rate_history.append(current_rate)

        # Detect slowdown
        if current_rate < RATE_THRESHOLD_LOW and self.slowdown_start is None:
            self.slowdown_start = time.time()
            logging.warning(f"Rate slowdown detected: {current_rate:.2f} Hz")

        # Detect recovery
        if current_rate > RATE_THRESHOLD_HIGH and self.slowdown_start is not None:
            recovery_time = time.time() - self.slowdown_start
            self.recovery_times.append(recovery_time)
            logging.info(
                f"Rate recovered after {recovery_time:.1f} seconds. Current rate: {current_rate:.2f} Hz")
            self.slowdown_start = None

    def get_rate(self):
        if len(self.timestamps) < 2:
            return 0.0
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        return (len(self.timestamps) - 1) / time_diff

    def get_recovery_stats(self):
        if not self.recovery_times:
            return None
        return {
            'min': min(self.recovery_times),
            'max': max(self.recovery_times),
            'avg': sum(self.recovery_times) / len(self.recovery_times),
            'count': len(self.recovery_times)
        }


# Thread-local storage for rate trackers
thread_local = threading.local()


def get_thread_rate_tracker():
    if not hasattr(thread_local, 'rate_tracker'):
        thread_local.rate_tracker = RateTracker(RATE_WINDOW)
    return thread_local.rate_tracker


def validate_cache_entry(line):
    """Validate a single cache entry line."""
    try:
        url, status = line.strip().split(',')
        if status not in ['R', 'U']:
            return False, f"Invalid status: {status}"
        # Basic URL validation
        if not url.startswith('https://www.wybory.gov.pl/'):
            return False, f"Invalid URL format: {url}"
        return True, None
    except ValueError:
        return False, f"Invalid line format: {line}"


def load_cache():
    cache = {}
    if not os.path.exists(CACHE_FILE):
        logging.info("No cache file found. Starting fresh.")
        return cache

    # First pass: validate all entries
    invalid_entries = []
    with open(CACHE_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            is_valid, error = validate_cache_entry(line)
            if not is_valid:
                invalid_entries.append((i, error))

    if invalid_entries:
        logging.error("Cache file contains invalid entries:")
        for line_num, error in invalid_entries:
            logging.error(f"Line {line_num}: {error}")
        raise ValueError("Cache file contains invalid entries")

    # Second pass: check for duplicates
    seen_urls = set()
    duplicates = []
    with open(CACHE_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            url, _ = line.strip().split(',')
            if url in seen_urls:
                duplicates.append(i)
            seen_urls.add(url)

    if duplicates:
        logging.error(
            f"Cache file contains {len(duplicates)} duplicate entries at lines: {duplicates}")
        raise ValueError("Cache file contains duplicate entries")

    # If we get here, the cache is valid. Load it.
    with open(CACHE_FILE, 'r') as f:
        for line in f:
            if line.strip():
                url, status = line.strip().split(',')
                cache[url] = status

    logging.info(f"Successfully loaded cache with {len(cache)} entries")
    return cache


def save_to_cache(url, status):
    with open(CACHE_FILE, 'a') as f:
        f.write(f"{url},{status}\n")


def parse_page_content(html_content):
    """Parse the HTML content and extract relevant information."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Initialize data structure
    data = {
        'district_number': None,
        'district_name': None,
        'total_voters': None,
        'total_cards': None,
        'valid_cards': None,
        'candidates': []
    }

    try:
        # Extract district information
        district_info = soup.find('div', class_='obkw')
        if district_info:
            title = district_info.find('h1', class_='title')
            if title:
                data['district_number'] = title.text.strip()

            # Get district name from info section
            info_section = district_info.find('dl', class_='info')
            if info_section:
                for dt, dd in zip(info_section.find_all('dt'), info_section.find_all('dd')):
                    if 'Nazwa' in dt.text:
                        data['district_name'] = dd.text.strip()

        # Extract voting statistics
        stats_table = district_info.find('table', class_='table')
        if stats_table:
            rows = stats_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].text.strip()
                    value = cells[1].text.strip().replace(' ', '')
                    if 'Uprawnionych' in label:
                        data['total_voters'] = int(value)
                    elif 'Wydanych' in label:
                        data['total_cards'] = int(value)
                    elif 'Ważnych' in label:
                        data['valid_cards'] = int(value)

        # Extract candidate information
        candidates_section = district_info.find('div', class_='can')
        if candidates_section:
            candidates_table = candidates_section.find('table', class_='table')
            if candidates_table:
                rows = candidates_table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        candidate_data = {
                            'name': cells[0].text.strip(),
                            'votes': int(cells[1].text.strip().replace(' ', '')),
                            'percentage': float(cells[2].text.strip().rstrip('%').replace(',', '.'))
                        }
                        data['candidates'].append(candidate_data)

    except Exception as e:
        logging.error(f"Error parsing page content: {str(e)}")
        return None

    return data


def save_results(results):
    """Save results to JSON file."""
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")


def check_url(url, cache, driver):
    # Check if URL is already in cache
    if url in cache:
        status = cache[url]
        logging.info(f"URL {url} found in cache with status: {status}")
        return status == 'R'

    try:
        # Get page content using Selenium
        html_content = get_page_content(url, driver)
        if not html_content:
            status = 'U'
            logging.warning(f"Failed to load content for URL {url}")
            save_to_cache(url, status)
            return {'status': status, 'data': None}

        # Get thread-specific rate tracker
        rate_tracker = get_thread_rate_tracker()
        rate_tracker.add_request()
        current_rate = rate_tracker.get_rate()

        # Extract just the number from the URL for cleaner logging
        url_number = url.split('/')[-1]
        logging.info(
            f"URL https://<...>/{url_number} is reachable, rate={current_rate:.2f}[hz]")

        # Parse the page content
        page_data = parse_page_content(html_content)
        if page_data:
            status = 'R'
            save_to_cache(url, status)
            return {'status': status, 'data': page_data}
        else:
            status = 'U'
            logging.warning(f"Failed to parse content for URL {url}")
            save_to_cache(url, status)
            return {'status': status, 'data': None}

    except Exception as e:
        status = 'U'
        logging.error(f"Error checking URL {url}: {str(e)}")
        save_to_cache(url, status)
        return {'status': status, 'data': None}


def increment_url_number(base_url, start_number, end_number, max_workers=10):
    # assert that start_number <= end_number
    assert start_number <= end_number, "start_number must be less than or equal to end_number"

    # Load existing cache
    cache = load_cache()

    # Parse the URL to get its components
    parsed_url = urlparse(base_url)
    path_parts = parsed_url.path.split('/')

    # Get the base path without the number
    base_path = '/'.join(path_parts[:-1])

    # Create a list of URLs to check
    urls_to_check = []
    numbers_checked = 0

    # Keep checking numbers until we hit our target count
    current_number = start_number
    while numbers_checked < (end_number - start_number):
        new_path = f"{base_path}/{current_number}"
        new_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            new_path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment
        ))

        # If URL is in cache, skip it
        if new_url in cache:
            current_number += 1
            continue

        # If not in cache, add to check list
        urls_to_check.append(new_url)
        numbers_checked += 1
        current_number += 1

    if not urls_to_check:
        logging.info("All numbers in range are already in cache")
        return []

    logging.info(f"Found {len(urls_to_check)} new URLs to check")

    # Create a WebDriver for each thread
    drivers = [setup_driver() for _ in range(max_workers)]

    try:
        # Use ThreadPoolExecutor to check URLs concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda args: check_url(args[0], cache, args[1]),
                zip(urls_to_check, drivers)))
    finally:
        # Clean up WebDrivers
        for driver in drivers:
            try:
                driver.quit()
            except:
                pass

    # Filter and save results
    valid_results = {url: result['data'] for url, result in zip(urls_to_check, results)
                     if result['status'] == 'R' and result['data'] is not None}
    if valid_results:
        save_results(valid_results)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check election URLs with rate monitoring')
    parser.add_argument('-n', '--number', type=int, default=32143,
                        help='Number of new entries to check (default: 32143)')
    parser.add_argument('-s', '--start', type=int, default=1467396,
                        help='Starting number for URL checking (default: 1467396)')
    parser.add_argument('-w', '--workers', type=int, default=10,
                        help='Number of worker threads (default: 10)')
    return parser.parse_args()


def main():
    args = parse_args()

    base_url = "https://www.wybory.gov.pl/prezydent2025/pl/obkw/2/1467396"
    start_number = args.start
    end_number = start_number + args.number

    logging.info(
        f"Starting URL check from {start_number} to {end_number} (checking {args.number} entries)")

    # Add a small delay between requests to be respectful to the server
    time.sleep(1)

    results = increment_url_number(
        base_url, start_number, end_number, max_workers=args.workers)

    # Count successful checks
    successful_checks = sum(1 for result in results if result['status'] == 'R')
    logging.info(f"Completed checking {len(results)} URLs")
    logging.info(f"Found {successful_checks} reachable URLs")

    # Print recovery statistics for each thread
    for i in range(args.workers):
        # Create a new tracker to access thread data
        thread_local.rate_tracker = RateTracker(RATE_WINDOW)
        stats = thread_local.rate_tracker.get_recovery_stats()
        if stats:
            logging.info(
                f"Thread {i} recovery stats: min={stats['min']:.1f}s, max={stats['max']:.1f}s, avg={stats['avg']:.1f}s, count={stats['count']}")


if __name__ == "__main__":
    main()
