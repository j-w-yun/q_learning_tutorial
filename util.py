import six


class Color:
	GRAY = 30
	RED = 31
	GREEN = 32
	YELLOW = 33
	BLUE = 34
	MAGENTA = 35
	CYAN = 36
	WHITE = 37
	CRIMSON = 38


COLOR_TO_NUM = dict(
	gray=Color.GRAY,
	red=Color.RED,
	green=Color.GREEN,
	yellow=Color.YELLOW,
	blue=Color.BLUE,
	magenta=Color.MAGENTA,
	cyan=Color.CYAN,
	white=Color.WHITE,
	crimson=Color.CRIMSON,
)


def colorize(string, color, bold=False, highlight=False):
	attr = []
	if isinstance(color, int):
		num = color
	else:
		num = COLOR_TO_NUM[color]

	if highlight:
		num += 10

	attr.append(six.u(str(num)))

	if bold:
		attr.append(six.u('1'))

	attrs = six.u(';').join(attr)
	return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)
