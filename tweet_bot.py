
import sched
import time
import os
import logging

from base_classes.reportGenerator import ReportGenerator
from twitter_bot_utilities.textToImage import text_to_image, image_to_media
from twitter_bot_utilities.bot_utils import get_twitter_client

try:
    os.mkdir("logs")
except FileExistsError:
    pass

def setup_logger(name, logfile_path, format):
    handler = logging.FileHandler(logfile_path)
    handler.setFormatter(format)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

# period = 21600 # time between posts in seconds
period = 10 # time between posts in seconds
client = get_twitter_client()
scheduler = sched.scheduler(time.time, time.sleep)
ufo_formatter = logging.Formatter('%(asctime)s - REPORT: %(message)s')
ufo_logger = setup_logger("UFO_LOGGER", os.path.join("logs", "ufo.log"), ufo_formatter)
gen_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
gen_logger = setup_logger("GENERAL", os.path.join("logs", "general.log"), gen_formatter)

def post_new_report_tweet():
    starttime = time.time()
    # Generate new report
    reportGenerator = ReportGenerator()
    report = reportGenerator.generate_report(1000, 0.1)
    ufo_logger.info(report)
    # Turn image into tweet
    img = text_to_image(report)
    media_ids = image_to_media(img)
    # client.create_tweet(media_ids=media_ids)
    return time.time() - starttime

def begin_scheduler():
    gen_logger.info("Beginning Report Scheduling Loop")
    while True:
        delta_time = post_new_report_tweet()
        gen_logger.info("Report generated in "+ f'{delta_time:.2f}' + " seconds.")
        print("Generated at time: ", time.time())
        if delta_time < period:
            time.sleep(period - delta_time)

if __name__ == "__main__":
    gen_logger.info("Starting UFO Sighting Report Tweet Bot")
    get_twitter_client()
    begin_scheduler()