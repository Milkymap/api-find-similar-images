from sys import stdout 
from loguru import logger 

log_format = [
	'<W><k>{time: YYYY-MM-DD hh:mm:ss}</k></W>',
	'<c>{file:<7}</c>',
	'<e>{line:03d}</e>',
	'<r>{level:^10}</r>',
	'<Y><k>{message}</k></Y>'
]

logger.remove()
logger.add(sink=stdout, level='TRACE', format='#'.join(log_format))