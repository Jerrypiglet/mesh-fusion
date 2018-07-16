import os
import argparse
import numpy as np
import sys

sys.path.insert(0,'..')
import common
from scale_utils import *
from fusion_utils import *
from simplify_utils import *

def get_parser():
    """
    Get parser of tool.

    :return: parser
    """

    parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
    parser.add_argument('--in_dir', type=str, default='../data/cars/0_in', help='Path to input directory.')
    parser.add_argument('--scale_dir', type=str, default='../data/cars/batch_scaled',help='Path to output directory; files within are overwritten!')
    parser.add_argument('--depth_dir', type=str, default='../data/cars/batch_depth', help='Path to output directory; files within are overwritten!')
    parser.add_argument('--out_dir', type=str, default='../data/cars/batch_watertight', help='Path to output directory; files within are overwritten!')
    parser.add_argument('--padding', type=float, default=0.1, help='Relative padding applied on each side.')

    parser.add_argument('--n_views', type=int, default=100, help='Number of views per model.')
    parser.add_argument('--image_height', type=int, default=640, help='Depth image height.')
    parser.add_argument('--image_width', type=int, default=640, help='Depth image width.')
    parser.add_argument('--focal_length_x', type=float, default=640, help='Focal length in x direction.')
    parser.add_argument('--focal_length_y', type=float, default=640, help='Focal length in y direction.')
    parser.add_argument('--principal_point_x', type=float, default=320, help='Principal point location in x direction.')
    parser.add_argument('--principal_point_y', type=float, default=320, help='Principal point location in y direction.')

    parser.add_argument('--depth_offset_factor', type=float, default=1.5, help='The depth maps are offsetted using depth_offset_factor*voxel_size.')
    parser.add_argument('--resolution', type=float, default=256, help='Resolution for fusion.')
    parser.add_argument('--truncation_factor', type=float, default=10, help='Truncation for fusion is derived as truncation_factor*voxel_size.')

    parser.add_argument('--log_scales', type=bool, default=True, help='If printing logging information for scales of meshes.')
    return parser

if __name__ == '__main__':

    parser = get_parser()
    options = parser.parse_args()

    scale_tools = Scale(options)
    fusion_tools = Fusion(options)

    assert os.path.exists(options.in_dir)
    common.makedir(options.scale_dir)
    common.makedir(options.depth_dir)
    common.makedir(options.out_dir)
    
    
    files_unfiltered = scale_tools.read_directory(options.in_dir)
    files = [file for file in files_unfiltered if '.off' in file]
    print('= Found %s OFFs in %s'%(len(files), options.in_dir))
    timer = common.Timer()
    Rs = fusion_tools.get_views()

    for idx, filepath in enumerate(files):
        print('=== Processing %d/%d OFFs...'%(idx+1, len(files)))
        """
        scale all found OFF files
        """
        mesh = common.Mesh.from_off(filepath)
        # mesh.switch_axes(0, 2)

        # Get extents of model.
        min, max = mesh.extents()
        if options.log_scales:
            print('[Data] %s extents before %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        # Set the center (although this should usually be the origin already).
        centers = (
            (min[0] + max[0]) / 2,
            (min[1] + max[1]) / 2,
            (min[2] + max[2]) / 2
        )
        # Scales all dimensions equally.
        sizes = (
            total_max - total_min,
            total_max - total_min,
            total_max - total_min
        )
        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        scales = (
            1 / (sizes[0] + 2 * scale_tools.options.padding * sizes[0]),
            1 / (sizes[1] + 2 * scale_tools.options.padding * sizes[1]),
            1 / (sizes[2] + 2 * scale_tools.options.padding * sizes[2])
        )

        mesh.translate(translation)
        mesh.scale(scales)

        min, max = mesh.extents()
        if options.log_scales:
            print('[Data] %s extents after scaled %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

        # May also switch axes if necessary.
        mesh.switch_axes(0, 2)

        scaled_off_path = os.path.join(options.scale_dir, os.path.basename(filepath))
        mesh.to_off(scaled_off_path)

        """
        Run rendering.
        """
        timer.reset()
        mesh = common.Mesh.from_off(scaled_off_path)
        depths = fusion_tools.render(mesh, Rs)

        depth_file_path = os.path.join(options.depth_dir, os.path.basename(filepath) + '.h5')
        common.write_hdf5(depth_file_path, np.array(depths))
        print('----- [Depth] wrote %s (%f seconds)' % (depth_file_path, timer.elapsed()))

        """
        Performs TSDF fusion.
        As rendering might be slower, we wait for rendering to finish.
        This allows to run rendering and fusing in parallel (more or less).
        """
        depths = common.read_hdf5(depth_file_path)

        timer.reset()
        tsdf = fusion_tools.fusion(depths, Rs)
        tsdf = tsdf[0]

        vertices, triangles = libmcubes.marching_cubes(-tsdf, 0)
        vertices /= options.resolution
        vertices -= 0.5

        # vertices = vertices[:, [2, 1, 0]]
        # print(vertices.shape)

        off_file = os.path.join(options.out_dir, ntpath.basename(filepath))
        # libmcubes.export_off(vertices, triangles, off_file)

        """
        Revert to original scales
        """
        mesh = common.Mesh.from_off(off_file)
        # mesh.switch_axes(0, 2)
        scales_back = [1./scale for scale in scales]
        mesh.scale(tuple(scales_back))
        translations_back = [-translation for translation in translation]
        mesh.translate(tuple(translations_back))
        mesh.to_off(off_file.replace('.off', '_ori_scale.off'))

        min, max = mesh.extents()
        if options.log_scales:
            print('[Data] %s extents scaled BACK %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

        print('----- [Fused Mesh] wrote %s (%f seconds)' % (off_file, timer.elapsed()))