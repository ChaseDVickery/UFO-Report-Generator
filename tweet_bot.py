
from base_classes.reportGenerator import ReportGenerator
from twitter_bot_utilities.textToImage import textToImage

# test_string = "This is the desired report jk kshdkfahk dsahfs hfkalsflk hasfh lksahflkj hslkjflksahf ahfkh slk toolongforonelinefhsakhflksahfksahkdjhfslkdshh hkf fshf"
# test_string += 9*test_string
# img = textToImage(test_string)

reportGenerator = ReportGenerator()
report = reportGenerator.generate_report(1000, 0.1)

img = textToImage(report)
img.save("testimg.png")