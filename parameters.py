# Tweak these parameters and see how the results change.
COLOR_SPACE = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 9  # HOG orientations
PIX_PER_CELL = 6  # HOG pixels per cell
CELL_PER_BLOCK = 2  # HOG cells per block
HOG_CHANNEL = "ALL"  # Can be 0, 1, 2, or "ALL"
SPATIAL_SIZE = (16, 16)  # Spatial binning dimensions
HIST_BINS = 16    # Number of histogram bins
SPATIAL_FEAT = True  # Spatial features on or off
HIST_FEAT = False  # Histogram features on or off
HOG_FEAT = True  # HOG features on or off
Y_START_STOP = [410, 630]  # Min and max in y to search in slide_window()
SCALE = 1.5
