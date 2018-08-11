def rect_to_bb(rect):
	"""
	In OpenCV, we normally think of a bounding box in terms
	of “(x, y, width, height)” so as a matter of convenience,
	the rect_to_bb  function takes this rect  object and transforms
	it into a 4-tuple of coordinates.

	This function accepts a single argument `rect`, which is
	assumed to be a bounding box rectangle produced by a
	dlib detector (i.e., the face detector).

	The rect  object includes the (x, y)-coordinates of the
	detection.
	"""	
	# take a bounding box predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype = "int"):
	"""	
	The dlib face landmark detector will return a `shape`  object
	containing the 68 (x, y)-coordinates of the facial landmark
	regions.

	Using the `shape_to_np` function, we cam convert this object
	to a NumPy array, allowing it to “play nicer” with our Python
	code.
	"""
	import numpy as np
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype = dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords