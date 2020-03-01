from numpy import matrix, transpose, asarray

"""
function that loads intrinsic and rotation Matrix 
and then returns them as one
"""


def create_transformation_matrix(calLines):
    intrinsic_matrix = matrix(calLines[0] + ";" + calLines[1] + ";" + calLines[2])
    rotation_matrix = matrix(calLines[6] + ";" + calLines[7] + ";" + calLines[8])
    transformation_matrix = intrinsic_matrix * rotation_matrix
    return transformation_matrix


"""
function that loads offset vector
"""


def create_offset_vec(calLines):
    offset_vec = matrix(calLines[10])
    return offset_vec


def get_faceCenter_cords(poseLines, calLines):
    depth_position_vec = matrix(poseLines[4])

    # offset of the RGB and depth camera
    offset_vec = create_offset_vec(calLines)
    rgb_position_vec = depth_position_vec + offset_vec

    # matrix that transforms depth data to RGB data
    transformation_matrix = create_transformation_matrix(calLines)
    rgb_position_vec = transformation_matrix * transpose(rgb_position_vec)

    # transfer from homogeneous cords to cartestian cords (x/z & y/z)
    face_cords = rgb_position_vec[:2] / rgb_position_vec[2]
    return face_cords[0].item(), face_cords[1].item()
