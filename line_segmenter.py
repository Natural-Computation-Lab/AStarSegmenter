import os, shutil
import numpy as np
import math
from itertools import groupby
from skimage import io
from skimage import img_as_ubyte
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.util import invert
from scipy.signal import find_peaks
from heapq import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
AstarSegmenter method for handwritten textline segemntation
"""

IN_FOLDER = "input_data\examples"
OUT_FOLDER = "lines"

SURNITA = False

# Nombers of rows
N_ROWS = {"moccia01.jpg":None, "santgall02.png":24, }  # None if no number of rows available
#N_ROWS = None # if no number of rows available

# Parameters
SIGMA = 10      #sigma for gaussian filter
NUM_OF_VERTICAL_ZONES = 8


# peak detection
TH_PEAK = 10
DISTANCE_PK = 35
PROMINANCE_PK = 100


# Constants ------------------
MAX_DISTANCE_SEGMENT = 50
SAVE_LINE_IMAGES = True
SAVE_LINEDOCSEG_IMAGE = True
ADD_FIRST_LAST_SEGMENT = True # add boundary to linesegmentation the top and bottob of the page detection

# PRINT
SHOW_PAGE_DET= False
SHOW_HISTO = False  

if os.path.exists(OUT_FOLDER):
    shutil.rmtree(OUT_FOLDER)
os.makedirs(OUT_FOLDER)

def horizontal_projections(image):
    return np.sum(image, axis=1) 


def vertical_projections(image):
    return np.sum(image, axis=0) 

def find_peak_regions(hpp, n_rows=0, divider=2):
    """
    find peak in orizontal projection
    if n_rows = 0, method returns all the peaks grater than max/divider
    else, nethod returns the top n_rows lines, if there are
    """

    if n_rows == 0:
        return _find_peak_regions_divider(hpp, divider=divider)
    else:
        return _find_peak_regions_nrows(hpp, n_rows)

def _find_peak_regions_divider(hpp, divider=2):
    """
    find peak in orizontal projection
    The “divider” parameter defaults to 2, which means the method will be thresholding the regions in 
    the middle of higher and lower peaks in the HPP.
    """
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks

def _find_peak_regions_nrows(hpp, n_rows):
    """
    find peak in orizontal projection
    The methods return the first top n_rows peaks, if they exist.
    """
    threshold = (np.max(hpp)-np.min(hpp))/TH_PEAK
    distance = (len(hpp))/DISTANCE_PK#(n_rows*2)
    prominence = (len(hpp))/PROMINANCE_PK#n_rows
    peaks = []
    pk_sorted = []

    all_peaks_ind, _ = find_peaks(hpp, height=threshold, distance=distance, prominence=prominence)

    if SHOW_HISTO:
        counts, bins = np.histogram(hpp)
        #plt.hist(bins[:-1], bins, weights=counts)
        #plt.stairs(hpp, hpp)
        #plt.stairs(counts, bins)
        plt.plot(hpp)
        plt.plot(all_peaks_ind, hpp[all_peaks_ind], "x", color="limegreen", ms=8, mew=2)
        plt.plot(np.zeros_like(hpp), "--", color="gray")
        plt.show()

    for pk_index in all_peaks_ind:
        pk_sorted.append([pk_index, hpp[pk_index]])
    
    pk_sorted.sort(key=lambda x: x[-1], reverse=True)

    row_founded = 0
    current_peak = 0
    while row_founded < n_rows and current_peak<len(pk_sorted): 
        pk_index, pk_value = pk_sorted[current_peak]
        if [pk_index, pk_value] not in peaks:
            row_founded += 1
            peaks.append([pk_index, pk_value])
            threshold = (pk_value-np.min(hpp))/2
            sx_index = pk_index-1
            dx_index = pk_index+1
            while hpp[sx_index] > threshold:
                peaks.append([sx_index, hpp[sx_index]])
                sx_index -= 1
            while hpp[dx_index] > threshold:
                peaks.append([dx_index, hpp[dx_index]])
                dx_index += 1
        current_peak += 1

    #peaks.sort(key=lambda x: x[0])
    
    peak_ret = []
    for i, hppv in enumerate(hpp):
        if [i, hppv] not in peaks:
            peak_ret.append([i, hppv])


    return peak_ret


def get_hpp_walking_regions(peaks_index):
    """
    group the peaks into walking windows
    """
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)

        if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []

        #get the last cluster
        if index == len(peaks_index)-1:
            hpp_clusters.append(cluster)
            cluster = []
            
    return hpp_clusters

def get_binary(img):
    """
    Binarize image with OTSU
    """
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary*1
    return binary

def path_exists(window_image):
    """
    very basic check first then proceed to A* check
    """
    if 0 in horizontal_projections(window_image):
        return True
    
    padded_window = np.zeros((window_image.shape[0],1))
    world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
    if len(path) > 0:
        return True
    
    return False

def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    
    for col in range(nmap.shape[1]):
        start = col
        end = col+20 # fized step window
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break
            
    return road_blocks

def get_road_block_regions_project(nmap):
    road_blocks = []
    needtobreak = False

    projection = vertical_projections(nmap)
    road_blocks_indices = np.where(projection != 0)[0]

    i=0
    while i < len(road_blocks_indices):
        start = road_blocks_indices[i]
        end = start
        while i+1<len(road_blocks_indices) and road_blocks_indices[i+1]-end == 1:
            i += 1
            end = road_blocks_indices[i]
        
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
             road_blocks.append([start, end])

        if needtobreak == True:
            break
        i += 1
            
    return road_blocks

def group_the_road_blocks(road_blocks):
    #group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

        if index == size-1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups

def get_page(image, margin=0.01, v=2, th=0):
    """
    Find the page in the image.
    the result is the part of the image that shows only text
    """
    vertical_p = vertical_projections(image)
    horizontal_p = horizontal_projections(image)

    m_v = int(image.shape[0]*margin)
    m_h = int(image.shape[1]*margin)

    max_v_p = int(max(vertical_p[m_v:image.shape[0]-m_v])/v)
    max_h_p = int(max(horizontal_p[m_h:image.shape[1]-m_h])/v)

    left = m_v
    while  vertical_p[left] <= max_v_p:
        left += 1
    while left > 0 and vertical_p[left] > th:
        left -= 1

    right = len(vertical_p)-1-m_v
    while  vertical_p[right] <= max_v_p:
        right -= 1
    while right < len(vertical_p) and vertical_p[right] > th:
        right += 1

    top = m_h
    while  horizontal_p[top] <= max_h_p:
        top += 1
    while top > 0 and horizontal_p[top] > th:
        top -= 1

    bottom = len(horizontal_p)-1-m_h
    while  horizontal_p[bottom] <= max_h_p:
        bottom -= 1
    while bottom <len(horizontal_p)-1 and horizontal_p[bottom] > th:
        bottom += 1

    top = top#-m_v
    right = right#+m_h
    left = left#-m_h
    bottom = bottom#+m_v

    return (top, right, left, bottom)

def _linesegmentation(crop_image, n_rows=0):
    """
    apply the line segmentation method to the crop_image.
    it returns a list of cut boundaries
    NOTE: the crop_image must be a BW image
    """
    inverse_image = gaussian(crop_image, sigma=SIGMA)
    th = threshold_otsu(inverse_image)
    inverse_image[inverse_image >= th] = 1
    inverse_image[inverse_image < th] = 0
    inverse_image = invert(inverse_image)

    hpp = horizontal_projections(inverse_image) 

    peaks = find_peak_regions(hpp, n_rows=n_rows)
    peaks_index = np.array(peaks)[:,0].astype(int)

    segmented_img = np.copy(crop_image)
    r,c = segmented_img.shape
    for ri in range(r):
        if ri in peaks_index:
            segmented_img[ri, :] = 0
            
    # compute the line cluster basing on oriz projection
    hpp_clusters = get_hpp_walking_regions(peaks_index)

    # refine binarization and add doorways  (when there is no path in the row division (total ascender or descender))
    binary_image = get_binary(crop_image)
    for cluster_of_interest in hpp_clusters:
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        if nmap.shape[0]>0:
            road_blocks_cluster_groups = get_road_block_regions_project(nmap)
            #create the doorways
            for index, road_blocks in enumerate(road_blocks_cluster_groups):
                window_offset = road_blocks[1]
                while window_offset < segmented_img.shape[1] and np.sum(nmap[:,window_offset], axis=0) != 0:
                    window_offset +=1
                window_image = nmap[:, road_blocks[0]: window_offset] # +1 OR NOT???
                binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: window_offset][int(window_image.shape[0]/2),:] *= 0
            
    #segment all the lines using the A* algorithm
    line_segments = []
    for i, cluster_of_interest in tqdm(enumerate(hpp_clusters)):   
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        if nmap.shape[0]>0:
            start = (int(nmap.shape[0]/2), 0)
            goal = (int(nmap.shape[0]/2),nmap.shape[1]-1)
            path = astar(nmap, start, goal)
            if len(path)>0:
                path.append((path[-1][0],0))
                path = np.array(path)
                offset_from_top = cluster_of_interest[0]
                path[:,0] += offset_from_top
                line_segments.append(path)
    
    ####################### view all line boundary
    # for path in line_segments:#original_line_segments
    #    plt.plot((path[:,1]), path[:,0], "r-")
    # plt.axis("off")
    # plt.imshow(crop_image, cmap="gray")
    # plt.show()

    return line_segments

def linesegmentation(image, n_rows=0, n_vertical_crops=1):
    """
    divides the image in vertical zones (according to the param n_vertical crops)
    and performs the line segmentationo in each zone
    """
    w = image.shape[1]
    w_zone = math.ceil(w/n_vertical_crops)

    all_zones_segments = []

    for itr in range(n_vertical_crops):
        end = (itr+1)*w_zone
        if itr == n_vertical_crops-1:
            end = w+1

        curr_v_zone = image[:, itr*w_zone:end]
        zone_segment = _linesegmentation(curr_v_zone, n_rows=n_rows)

        all_zones_segments.append(zone_segment)

    # balance boundaries number in zones
    all_zones_segments = _balance_segments(all_zones_segments, w_zone)

    # Fuse all zones in one list of boundaries
    line_segments = []
    for itr_1 in range(len(all_zones_segments[0])):
        line_segment = all_zones_segments[0][itr_1]
        for itr_2 in (range(1,n_vertical_crops)):
            line_segment_to_merge = _find_tomerge_segment(line_segment, all_zones_segments[itr_2], w_zone)
            line_segment = _merge_segments(line_segment, line_segment_to_merge)
        line_segments.append(line_segment)

    # sort line_segments   
    line_segments = _sort_segments_list(line_segments)

    return line_segments

def _sort_segments_list(line_segments):
    sorted_list= []
    tup_segments = []

    for el in line_segments:
        tup_segments.append((el[0][0], el))
    
    tup_segments.sort(key=lambda el: el[0])

    for el in tup_segments:
        sorted_list.append(el[-1])
    
    return sorted_list

def _merge_segments(base_segment, tomerge_segment):
    for point in reversed(tomerge_segment):
        h_val = np.array([point[0],base_segment[0][-1]+1], dtype=np.int32)
        base_segment = np.insert(base_segment, 0, h_val, axis=0)

    return base_segment

def _find_closest_segment(line_segment, next_zone_segments, next_dx=True):
    """
    retrun segment in next zone closer to the line_segment
    """
    if next_dx:
        pos = 0
        pos_test=-1
    else:
        pos = -1
        pos_test=0

    h_value = line_segment[pos][0]
    dist = float('inf')
    ret_segment = None
    for segment in next_zone_segments:
        test_h_value = segment[pos_test][0]
        if abs(h_value-test_h_value)<dist:
            dist = abs(h_value-test_h_value)
            ret_segment = segment
    return ret_segment, dist

def _find_tomerge_segment(line_segment, next_zone_segments, w_zone,  max_dist=MAX_DISTANCE_SEGMENT):
    """
    retrun segment in next zone closer to the line_segment.
    if there is no segment in the range max_dist, it returns a new segment at zero distance to line_segment
    """
    ret_segment, dist = _find_closest_segment(line_segment, next_zone_segments)
    
    # if nezt zonew boundary to much distant
    if dist > max_dist:
        h_value = line_segment[0][0]
        ret_segment = np.flip(np.column_stack(((np.ones((w_zone,))*h_value), np.arange(w_zone))).astype(int), axis=0)

    return ret_segment

def _balance_segments(all_zones_segments, w_zone):
    """
    define a new set of segments for each zone.
    the function return a set of boundaries zones all with the same number of boundaries
    """
    new_all_zones_segments  = []

    lens = []
    for zone in all_zones_segments:
        lens.append(len(zone))
    n_boundary = max(lens)

    for itr in range(len(all_zones_segments)-1):
        zone = all_zones_segments[itr] 
        next_zone = all_zones_segments[itr+1] 
        new_zone = []

        #remove isolate segments
        for segment in zone:
            _, dist = _find_closest_segment(segment, next_zone)
            if dist <= MAX_DISTANCE_SEGMENT:
                new_zone.append(segment)
         
        new_all_zones_segments.append(new_zone)
    
    new_all_zones_segments.append(all_zones_segments[-1])

    # add new segments to complete the line boundary
    w_itr = 0
    while w_itr<3 and not all_equal_lens(new_all_zones_segments):
        for itr in range(len(new_all_zones_segments)-1):
            zone = new_all_zones_segments[itr] 
            next_zone = new_all_zones_segments[itr+1]

            for segment in next_zone:
                _, dist = _find_closest_segment(segment, zone, next_dx=False)
                if dist > MAX_DISTANCE_SEGMENT:
                    h_value = segment[-1][0]
                    new_segment = np.flip(np.column_stack(((np.ones((w_zone,))*h_value), np.arange(w_zone))).astype(int), axis=0)
                    zone.append(new_segment)
        w_itr += 1

    return new_all_zones_segments

def all_equal_lens(all_zones_segments):
    """
    Returns True id all the zones contain the same number of segments
    """
    lens = []
    for zone in all_zones_segments:
        lens.append(len(zone))
    g = groupby(lens)
    return next(g, True) and not next(g, False)



#a star path planning algorithm 
def heuristic_SED(a, b):
    """
    Squared Euclidean Distance
    """
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

#a star path planning algorithm 
def heuristic_manhattan(a, b):
    """
    Manhattan distance
    """
    return abs((b[0] - a[0])) + abs((b[1] - a[1]))

def astar(array, start, goal, heuristic=heuristic_SED):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return []

def extract_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.max(upper_line[:, 0])
    img_copy = np.copy(image)
    #r, c = img_copy.shape
    for index in range(img_copy.shape[1]-1):
        img_copy[0:lower_line[index, 0], lower_line[index, 1]] = 255
        img_copy[upper_line[index, 0]:img_copy.shape[0], upper_line[index, 1]] = 255
    
    return img_copy[lower_boundary:upper_boundary, :]

def extract_line_from_image2(image, lower_line, upper_line):
    """
    extract line from boundary set
    """
    r, c = image.shape

    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    imr_row = np.ones((upper_boundary-lower_boundary, c), dtype=np.uint8)*255
   
    for index in range(c-1):
        col = image[lower_line[index, 0]:upper_line[index, 0], index]
        col_start = 0
        col_len = len(col)
        imr_row[col_start:col_len, index] = col # offset!
    
    return imr_row

def save_line_images(line_segments, image, out_folder):
    """
    Saves all images lines. It generates one image for each line detected.
    """
    line_images = []
    line_count = len(line_segments)
    for line_index in range(line_count-1):
        line_image = extract_line_from_image(image, line_segments[line_index], line_segments[line_index+1])
        line_images.append(line_image)
        io.imsave(os.path.join(out_folder,str(line_index+1).zfill(len(str(line_count)))+"."+image_name.split(".")[-1]), line_image)

def save_line_on_doc_image(line_segments, image, out_folder, color=(255,0,0), thickness=1):
    """
    Saves an image of the entire document page showing the cutting boundaries
    """
    out_image = rgb2gray(image)
    out_image = gray2rgb(out_image)
    out_image = img_as_ubyte(out_image)

    for line in line_segments:
        for x, y in line:
            if y >= image.shape[1]:
                y=image.shape[1]-1
            out_image[x:x+thickness-1,y,:] = color

    io.imsave(os.path.join(out_folder,"_all_lines."+image_name), out_image)



if __name__ == "__main__":
    for image_name in tqdm(os.listdir(IN_FOLDER)):
        curr_out_folder = os.path.join(OUT_FOLDER, image_name.split(".")[0])
        os.mkdir(curr_out_folder)

        image = io.imread(os.path.join(IN_FOLDER, image_name), as_gray=True)        
        image_orig = gray2rgb(image)

        inverse_image = gaussian(image, sigma=SIGMA)
        th = threshold_otsu(inverse_image)
        inverse_image[inverse_image >= th] = 1
        inverse_image[inverse_image < th] = 0
        inverse_image = invert(inverse_image)

        top, right, left, bottom = get_page(inverse_image)

        
        # line segmentation
        if N_ROWS is None:
            n_of_rows = 0
        else:
            n_of_rows = N_ROWS[image_name]
            if n_of_rows is None:
                n_of_rows = 0
        
        # Surinta method
        if SURNITA:
            top, right, left, bottom = 0, image.shape[1], 0, image.shape[0]
            NUM_OF_VERTICAL_ZONES = 1
            n_of_rows = 0


        # ## Print Page Detected ########################################
        #print(top, right, left, bottom)
        if SHOW_PAGE_DET:
            plt.imshow(image, cmap="gray")
            plt.gca().add_patch(Rectangle((left, top), right-left, bottom-top, edgecolor='red', facecolor='none', lw=1))
            plt.show()


        line_segments = linesegmentation(image[top:bottom, left:right], n_rows=n_of_rows, n_vertical_crops=NUM_OF_VERTICAL_ZONES)
       
        # compute borders for full image
        original_line_segments = []
        if ADD_FIRST_LAST_SEGMENT:
            #add first row
            first_top_row = np.flip(np.column_stack(((np.ones((image.shape[1],))*top), np.arange(image.shape[1]))).astype(int), axis=0)
            original_line_segments.append(first_top_row)
        for segment in line_segments:
            ref_right = segment[0][0]
            ref_left = segment[-1][0]
            original_segment = []
            for ind in range(image.shape[1]-right):
                original_segment.append(np.array([segment[0][0]+top, image.shape[1]-ind-1], dtype=np.int32))
            
            for point in segment:
                original_segment.append(np.array([point[0]+top, point[1]+left], dtype=np.int32))
            
            for ind in reversed(range(left)):
                original_segment.append(np.array([segment[-1][0]+top, ind+1], dtype=np.int32))

            original_line_segments.append( np.array(original_segment, dtype=np.int32))

        ####################### view all line boundary
        # fig = plt.figure(figsize=(1, 1))
        # for path in original_line_segments:#original_line_segments
        #     plt.plot((path[:,1]), path[:,0], "r-")
        # plt.axis("off")
        # plt.imshow(image, cmap="gray")
        # plt.show()
        #fig.savefig(os.path.join(OUT_FOLDER, "seg_"+image_name))

        if ADD_FIRST_LAST_SEGMENT:
            ## add an extra line to the line segments array which represents the last bottom row on the image
            last_bottom_row = np.flip(np.column_stack(((np.ones((image.shape[1],))*bottom), np.arange(image.shape[1]))).astype(int), axis=0)
            original_line_segments.append(last_bottom_row)

        ## Save images
        if SAVE_LINE_IMAGES:
            save_line_images(original_line_segments, image, curr_out_folder)
        if SAVE_LINEDOCSEG_IMAGE:
            save_line_on_doc_image(original_line_segments, image_orig, curr_out_folder, thickness=5)


    print("Done")