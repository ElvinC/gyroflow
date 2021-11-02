import os
import sys
from stabilizer import MultiStabilizer
from gyrolog import guess_log_type_from_video, guess_log_type_from_log
import smoothing_algos
import argparse

STATIC_ZOOM = 1/1 #TODO: get from argv
SPLIT_SCREEN = False
PREVIEW = False

parser = argparse.ArgumentParser(description='Gyroflow CLI')

group_input = parser.add_argument_group('Input')
group_input.add_argument('input', help='The video file to stabilize')
group_input.add_argument('--gyro', help='The file containing the gyro data')

group_stab = parser.add_argument_group('Stabilization')
group_stab.add_argument('--smooth', '-s', help='Smoothnes, in seconds', type=float, default=0.2)
group_stab.add_argument('--zoom', '-z', help='Zoom time, in seconds', type=float, default=5)
group_stab.add_argument('--camera', '-c', help='Path to camera preset', type=str, required=True) #TODO: automatically detect gopro camera

group_sync = parser.add_argument_group('Synchronization')
group_sync.add_argument('--auto-sync', help='Find sync points automatically', type=int, nargs='?', const=9, default=0)
group_sync.add_argument('--auto-sync-error-margin', help='Sync error margin (seconds)', type=float, default=0.02)
group_sync.add_argument('--sync-search-size', help='Sync search size (seconds)', type=float, default=10)
group_sync.add_argument('--sync-analyze-frames', help='Number of frames to analyse for each sync point', type=int, default=60)
group_sync.add_argument('--sync-analyze-frames-skip', help='Analyze every N frame. Speed up sync by skipping frames', type=int, default=1)
group_sync.add_argument('--sync', help='Add sync point manually (timestamp in seconds)', type=float, action='append', default=[])
group_sync.add_argument('--gyro-offset', help='Initial rough gyro offset in seconds', type=float, default=0.0)

group_output = parser.add_argument_group('Output')
group_output.add_argument('--no-audio', help='Export video without audio', type=bool, dest='audio', nargs='?', const=False, default=True) #TODO: Use `action=argparse.BooleanOptionalAction` (not available for some reason)
group_output.add_argument('--resolution', help='Render resolution. Examples: 1920x1080, 720p, 4k', type=str, default='auto') #TODO: should use camera preset to determine output resolution
group_output.add_argument('--bitrate', help='Render bitrate in Mbps', type=float, default=20) #TODO: calculate based on input bitrate and --resolution
group_output.add_argument('--vcodec', help="Video codec", choices=['libx264', 'h264_nvenc', 'h264_amf', 'h264_vaapi', 'h264_videotoolbox', 'prores_ks', 'v210'], default='libx264')
group_output.add_argument('--vprofile', help="Codec profile", type=str, default='main') # too many to list up choices here. TODO: different defaults for each vcodec.
group_output.add_argument('-y', help='Ignore if the output file already exists', type=bool, nargs='?', const=True, default=False) #TODO: Use `action=argparse.BooleanOptionalAction` (not available for some reason)
group_output.add_argument('-f', help='Output type (to output .gyroflow files)', choices=['video', 'gyroflow'], default='video')
group_output.add_argument('--out', '-o', help='Render stabilized video to this file')

group_trim = parser.add_argument_group('Trim')
group_trim.add_argument('--start', help='Start time', type=float, default=0)
group_trim.add_argument('--end', help='End time', type=float) # defaults to end of video

parser.add_argument_group()

args = parser.parse_args()

#
# Set some args automatically
#

if args.gyro is None:
    # Assuming gopro or insta360 where gyro is embedded in video file
    #TODO: look for *.gyroflow or .bbl or other log files
    args.gyro = args.input

if args.out is None:
    input_basename = os.path.basename(args.input)
    input_directory = os.path.dirname(args.input)
    input_basename_split = os.path.splitext(input_basename)
    args.out = os.path.join(input_directory, input_basename_split[0] + '.gyroflow.mp4')

if args.resolution == 'auto':
    #TODO: Auto should be as big as original; keep the largest dimension. Calculate the shorter one based on distortion? Or simply use 16:9?
    args.resolution = (1920, 1080)
elif args.resolution[-1].lower() == 'p': # 1080p, 720p resolution
    ASPECT_RATIO = 16/9
    height = int(args.resolution[0:-1])
    width = int(height * ASPECT_RATIO)
    args.resolution = (width, height)
elif args.resolution[-1].lower() == 'k': # 4k, 6k, 8k resolution
    k = float(args.resolution[0:-1])

    # Assuming UHD 16:9 (not DCI, 2.39:1)
    # 8k = 4320p
    # 4k = 2160p
    # 2k = 1080p
    # 1k = 540p

    height = int(540 * k)
    width = int(960 * k)
    args.resolution = (width, height)
else:
    args.resolution = tuple(map(int, args.resolution.split('x')[0:2]))

assert type(args.resolution) is tuple and len(args.resolution) == 2, 'invalid --resolution'

if args.f != 'video':
    #TODO: support .mp4.gyroflow export
    print('-f not implemented')
    exit(1)

print(args.input, args.out, args.smooth, args.zoom)

if not args.y and os.path.isfile(args.out):
    print('Output file already exists: {}'.format(args.out))
    exit(1)

# Initialize stabilization
guessed_log, logtype, logvariant = guess_log_type_from_video(args.gyro) # TODO: what about guess_log_type_from_log()?
stab = MultiStabilizer(args.input, args.camera, args.gyro, logtype=logtype, logvariant=logvariant)
duration = stab.num_frames / stab.fps

# Get timestamp where negative numbers count from end of video
normalize_timestamp = lambda time: time if time >= 0 else time + duration

stab.set_smoothing_algo(smoothing_algos.PlainSlerp()) #TODO: get from argv
stab.smoothing_algo.set_user_option('smoothness', args.smooth) #TODO: get from argv - let users set any smoothing algo option

if len(args.sync) or args.auto_sync > 0: # Gyro sync
    stab.multi_sync_init()
    stab.set_initial_offset(args.gyro_offset)
    stab.set_rough_search(args.sync_search_size)
    stab.set_num_frames_skipped(args.sync_analyze_frames_skip)

    # Array of [frame_number, frames_to_analyze]
    sync_points = [[normalize_timestamp(sync_point) * stab.fps, args.sync_analyze_frames] for sync_point in args.sync]
    
    # Sync
    success = stab.full_auto_sync_parallel(
        max_fitting_error = args.auto_sync_error_margin,
        max_points=args.auto_sync,
        n_frames=args.sync_analyze_frames,
        sync_points=sync_points,
        debug_plots=False
    )

    if not success:
        print(f'GYRO SYNC FAILED')
        exit(1)

elif type(stab.orientations) != type(None):  # no sync (GoPro 8, 9, 10)
    stab.update_smoothing()

else:
    print('You must use --auto-sync or provide sync points manully using --sync <time>')
    exit(1)

args.start = normalize_timestamp(args.start)
args.end = normalize_timestamp(args.end) if args.end is not None else duration

assert duration > args.start >= 0, '--start is invalid.'
assert duration >= args.end >= 0, '--end is invalid.'
assert args.start < args.end, '--start must a lower number than --end'

stab.renderfile(args.start, args.end, args.out,
    out_size = args.resolution,
    bitrate_mbits=args.bitrate,
    vcodec=args.vcodec,
    vprofile=args.vprofile,
    audio=args.audio,

    smoothingFocus=args.zoom,

    fov_scale=STATIC_ZOOM,
    split_screen=SPLIT_SCREEN,
    display_preview=PREVIEW,
)

print('Saved as {}'.format(args.out))
