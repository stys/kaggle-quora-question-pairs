import logging
from pyhocon import ConfigFactory

conf = ConfigFactory.parse_file('application.conf')
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S')
