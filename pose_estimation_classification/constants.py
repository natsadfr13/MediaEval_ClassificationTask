# CONSTANTS

STROKE_TO_CLASS = {'Serve Forehand Backspin': 0,
                  'Serve Forehand Loop': 1,
                  'Serve Forehand Sidespin': 2,
                  'Serve Forehand Topspin': 3,

                  'Serve Backhand Backspin': 4,
                  'Serve Backhand Loop': 5,
                  'Serve Backhand Sidespin': 6,
                  'Serve Backhand Topspin': 7,

                  'Offensive Forehand Hit': 8,
                  'Offensive Forehand Loop': 9,
                  'Offensive Forehand Flip': 10,

                  'Offensive Backhand Hit': 11,
                  'Offensive Backhand Loop': 12,
                  'Offensive Backhand Flip': 13,

                  'Defensive Forehand Push': 14,
                  'Defensive Forehand Block': 15,
                  'Defensive Forehand Backspin': 16,

                  'Defensive Backhand Push': 17,
                  'Defensive Backhand Block': 18,
                  'Defensive Backhand Backspin': 19,
                  
                  'Negative': 20}

CLASS_TO_STROKE = {0: 'Serve Forehand Backspin',
                  1: 'Serve Forehand Loop',
                  2: 'Serve Forehand Sidespin',
                  3: 'Serve Forehand Topspin',

                  4: 'Serve Backhand Backspin',
                  5: 'Serve Backhand Loop',
                  6: 'Serve Backhand Sidespin',
                  7: 'Serve Backhand Topspin',

                  8: 'Offensive Forehand Hit',
                  9: 'Offensive Forehand Loop',
                  10: 'Offensive Forehand Flip',

                  11: 'Offensive Backhand Hit',
                  12: 'Offensive Backhand Loop',
                  13: 'Offensive Backhand Flip',

                  14: 'Defensive Forehand Push',
                  15: 'Defensive Forehand Block',
                  16: 'Defensive Forehand Backspin',

                  17: 'Defensive Backhand Push',
                  18: 'Defensive Backhand Block',
                  19: 'Defensive Backhand Backspin',
                  
                  20: 'Negative'}

NB_STROKES = len(STROKE_TO_CLASS.keys())

## --- Paths to data ---
PATH_TO_DATA_POSE_ESTIMATION = './dataSetCSV'
PATH_TO_MODELS = "./models/"

MAX_SIZE_STROKE = 200 

NORMALIZATION_COLUMN = "1_x"

## --- Time sampling
WANTED_SAMPLING_TIMESTEP = 0.01 #s

SELECTED_FEATURES = ["box_conf", "box_x_n", "box_y_n", "box_w_n", "box_h_n",
                  "1_conf","1_x_n", "1_y_n",
                  "2_conf", "2_x_n", "2_y_n",
                  "3_conf", "3_x_n", "3_y_n",
                  "4_conf", "4_x_n", "4_y_n",
                  "5_conf", "5_x_n", "5_y_n",
                  "6_conf", "6_x_n", "6_y_n",
                  "7_conf", "7_x_n", "7_y_n",
                  "8_conf", "8_x_n", "8_y_n",
                  "9_conf", "9x_n", "9_y_n",
                  "10_conf", "10_x_n", "10_y_n",
                  "11_conf", "11_x_n", "11_y_n",
                  "12_conf", "12_x_n", "12_y_n",
                  "13_conf", "13_x_n", "13_y_n",
                  "14_conf", "14_x_n", "14_y_n",
                  "15_conf", "15_x_n", "15_y_n",
                  "16_conf", "16_x_n", "16_y_n",
                  "17_conf", "17_x_n", "17_y_n",
            ]

# Hands
# [
#                   "5_conf", "5_x_n", "5_y_n",
#                   "6_conf", "6_x_n", "6_y_n",
#                   "7_conf", "7_x_n", "7_y_n",
#                   "8_conf", "8_x_n", "8_y_n",
#                   "9_conf", "9x_n", "9_y_n",
#                   "10_conf", "10_x_n", "10_y_n",
#                   "11_conf", "11_x_n", "11_y_n",
#                   "12_conf", "12_x_n", "12_y_n",
#             ]

# All features 
# ["box_conf", "box_x", "box_y", "box_w", "box_h", "box_x_n", "box_y_n", "box_w_n", "box_h_n",
#       "1_x", "1_y", "1_conf", "1_x_n", "1_y_n",
#       "2_x", "2_y", "2_conf", "2_x_n", "2_y_n",
#       "3_x", "3_y", "3_conf", "3_x_n", "3_y_n",
#       "4_x", "4_y", "4_conf", "4_x_n", "4_y_n",
#       "5_x", "5_y", "5_conf", "5_x_n", "5_y_n",
#       "6_x", "6_y", "6_conf", "6_x_n", "6_y_n",
#       "7_x", "7_y", "7_conf", "7_x_n", "7_y_n",
#       "8_x", "8_y", "8_conf", "8_x_n", "8_y_n",
#       "9_x", "9_y", "9_conf", "9x_n", "9_y_n",
#       "10_x", "10_y", "10_conf", "10_x_n", "10_y_n",
#       "11_x", "11_y", "11_conf", "11_x_n", "11_y_n",
#       "12_x", "12_y", "12_conf", "12_x_n", "12_y_n",
#       "13_x", "13_y", "13_conf", "13_x_n", "13_y_n",
#       "14_x", "14_y", "14_conf", "14_x_n", "14_y_n",
#       "15_x", "15_y", "15_conf", "15_x_n", "15_y_n",
#       "16_x", "16_y", "16_conf", "16_x_n", "16_y_n",
#       "17_x", "17_y", "17_conf", "17_x_n", "17_y_n",
# ]

# All features non normalized
# ["box_conf", "box_x", "box_y", "box_w", "box_h",
#                 "1_x", "1_y", "1_conf",
#                 "2_x", "2_y", "2_conf",
#                 "3_x", "3_y", "3_conf",
#                 "4_x", "4_y", "4_conf",
#                 "5_x", "5_y", "5_conf",
#                 "6_x", "6_y", "6_conf", 
#                 "7_x", "7_y", "7_conf", 
#                 "8_x", "8_y", "8_conf", 
#                 "9_x", "9_y", "9_conf", 
#                 "10_x", "10_y", "10_conf", 
#                 "11_x", "11_y", "11_conf", 
#                 "12_x", "12_y", "12_conf", 
#                 "13_x", "13_y", "13_conf", 
#                 "14_x", "14_y", "14_conf", 
#                 "15_x", "15_y", "15_conf", 
#                 "16_x", "16_y", "16_conf", 
#                 "17_x", "17_y", "17_conf", 
#             ]

#All features normalized 
# ["box_conf", "box_x_n", "box_y_n", "box_w_n", "box_h_n",
#                   "1_conf","1_x_n", "1_y_n",
#                   "2_conf", "2_x_n", "2_y_n",
#                   "3_conf", "3_x_n", "3_y_n",
#                   "4_conf", "4_x_n", "4_y_n",
#                   "5_conf", "5_x_n", "5_y_n",
#                   "6_conf", "6_x_n", "6_y_n",
#                   "7_conf", "7_x_n", "7_y_n",
#                   "8_conf", "8_x_n", "8_y_n",
#                   "9_conf", "9x_n", "9_y_n",
#                   "10_conf", "10_x_n", "10_y_n",
#                   "11_conf", "11_x_n", "11_y_n",
#                   "12_conf", "12_x_n", "12_y_n",
#                   "13_conf", "13_x_n", "13_y_n",
#                   "14_conf", "14_x_n", "14_y_n",
#                   "15_conf", "15_x_n", "15_y_n",
#                   "16_conf", "16_x_n", "16_y_n",
#                   "17_conf", "17_x_n", "17_y_n",
#             ]

SELECTED_TRICKS = ['Serve Forehand Backspin',
                'Serve Forehand Loop',
                'Serve Forehand Sidespin',
                'Serve Forehand Topspin',

                'Serve Backhand Backspin',
                'Serve Backhand Loop',
                'Serve Backhand Sidespin',
                'Serve Backhand Topspin',

                'Offensive Forehand Hit',
                'Offensive Forehand Loop',
                'Offensive Forehand Flip',

                'Offensive Backhand Hit',
                'Offensive Backhand Loop',
                'Offensive Backhand Flip',

                'Defensive Forehand Push',
                'Defensive Forehand Block',
                'Defensive Forehand Backspin',

                'Defensive Backhand Push',
                'Defensive Backhand Block',
                'Defensive Backhand Backspin']

SELECTED_FEATURES_PLOT = [
                  "5_conf", "5_x_n", "5_y_n",
                  "6_conf", "6_x_n", "6_y_n",
                  "7_conf", "7_x_n", "7_y_n",
                  "8_conf", "8_x_n", "8_y_n",
                  "9_conf", "9x_n", "9_y_n",
                  "10_conf", "10_x_n", "10_y_n",
                  "11_conf", "11_x_n", "11_y_n",
                  "12_conf", "12_x_n", "12_y_n",
            ]