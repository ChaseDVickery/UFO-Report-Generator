
import sched
import time

from base_classes.reportGenerator import ReportGenerator
from twitter_bot_utilities.textToImage import text_to_image, image_to_media
from twitter_bot_utilities.bot_utils import get_twitter_client

# period = 21600 # time between posts in seconds
period = 10 # time between posts in seconds
client = get_twitter_client()
scheduler = sched.scheduler(time.time, time.sleep)

def post_new_report_tweet():
    starttime = time.time()
    # Generate new report
    reportGenerator = ReportGenerator()
    report = reportGenerator.generate_report(1000, 0.1)
    # Turn image into tweet
    img = text_to_image(report)
    media_ids = image_to_media(img)
    # client.create_tweet(media_ids=media_ids)
    print(report)
    return time.time() - starttime

def begin_scheduler():
    print("beginning scheduler")
    while True:
        delta_time = post_new_report_tweet()
        print(delta_time)
        print("Curr time: ", time.time())
        if delta_time < period:
            time.sleep(period - delta_time)

if __name__ == "__main__":
    begin_scheduler()