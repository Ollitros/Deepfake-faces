import numpy as np
import dlib
import cv2
import os
from imutils import face_utils


# Read points from text file
def readPoints(path):
    # Create an array of points.
    points = []

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def make_swap(src, dst, padding):

    img1Warped = np.copy(dst)

    # Set dlib parameters
    predictor_path = 'data/face_features/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # detect face src
    rects = detector(src, 1)
    print('rects - ', rects)

    try:
        roi = rects[0]  # region of interest
    except IndexError:
        return 'fiasko'
    shape = predictor(src, roi)
    # shape = src[y:y + h, x:x + w]
    shape = face_utils.shape_to_np(shape)
    np.savetxt('data/src.txt', shape, fmt="%s")

    # detect face dst
    rects = detector(dst, 1)
    print('rects - ', rects)

    try:
        roi = rects[0]  # region of interest
    except IndexError:
        return 'fiasko'

    shape = predictor(dst, roi)
    # shape = dst[y:y + h, x:x + w]
    shape = face_utils.shape_to_np(shape)
    np.savetxt('data/dst.txt', shape, fmt="%s")

    # Read array of corresponding points
    points1 = readPoints('data/src.txt')
    points2 = readPoints('data/dst.txt')

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # Find delanauy traingulation for convex hull points
    sizeImg2 = dst.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(src, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(dst.shape, dtype=dst.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    # Extracts face by mask
    # Clone seamlessly.
    # mask_out = cv2.subtract(mask, img1Warped)
    # mask_out = cv2.subtract(mask, mask_out)

    output = cv2.seamlessClone(np.uint8(img1Warped), dst, mask, center, cv2.NORMAL_CLONE)
    shape = output.shape
    output = output[padding: shape[0] - padding, padding: shape[1] - padding]

    return output


def main(path_to_frames, path_to_faces, path_to_faces_info, path_to_predictions):

    _, _, src_files = next(os.walk(path_to_frames))
    file_count = len(src_files)

    for i in range(file_count):
        print(i)
        index = src_files[i]
        index = index.split('.')
        index = index[0].split('frame')
        index = int(index[1])

        # Read src face
        src = cv2.imread((path_to_src_faces + 'src_face{i}.jpg').format(i=index))
        src = cv2.resize(src, (200, 200))

        # Read dst face
        prediction = cv2.imread((path_to_predictions + 'prediction{i}.jpg').format(i=index))
        prediction = cv2.resize(prediction, (200, 200))

        # Read frame
        frame = cv2.imread((path_to_frames + 'src_frame{i}.jpg').format(i=index))

        # Read src info
        info = None
        with open((path_to_faces_info + 'src_info{i}.txt').format(i=index), "r") as file:
            info = file.readline()
            info = info.split(" ")
            info = [int(info[0]), int(info[1]), int(info[2]), int(info[3])]
            print(info)

        # make paddings
        padding = 20
        prediction = cv2.copyMakeBorder(prediction, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0,0,0])
        src = cv2.copyMakeBorder(src, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Make face swap
        swaped_face = make_swap(src=prediction, dst=src, padding=padding)

        if swaped_face == "fiasko":
            print("fiasko")
        else:
            # Incertion process
            swaped_face = cv2.resize(swaped_face, (info[2], info[3]))
            frame[info[1]: info[1] + info[3], info[0]: info[0] + info[2]] = swaped_face
            cv2.imwrite('data/swapped_frames/swapped_frame{i}.jpg'.format(i=index), frame)


if __name__ == '__main__':
    path_to_frames = 'data/src/src_video_faces/frames/'
    path_to_src_faces = 'data/src/src_video_faces/faces/face_images/'
    path_to_src_faces_info = 'data/src/src_video_faces/faces/face_info/'
    path_to_predictions = 'data/predictions/'

    main(path_to_frames=path_to_frames, path_to_faces=path_to_src_faces,
         path_to_faces_info=path_to_src_faces_info, path_to_predictions=path_to_predictions)