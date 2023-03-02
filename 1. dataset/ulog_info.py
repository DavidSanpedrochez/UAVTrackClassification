from __future__ import print_function
import argparse
from pyulog.core import ULog


# https://github.com/PX4/pyulog/blob/main/pyulog/info.py
# https://docs.px4.io/main/en/dev_log/ulog_file_format.html
# https://px4.io/px4-uorb-explained-part-4-ulog-flight-logging-system/

# EJEMPLO: https://logs.px4.io/plot_app?log=c1060b4e-b930-43f2-a3ee-d5ca5a41b686

##################
# 
def show_info(ulog, verbose):
    """Show general information from an ULog"""
    m1, s1 = divmod(int(ulog.start_timestamp/1e6), 60)
    h1, m1 = divmod(m1, 60)
    m2, s2 = divmod(int((ulog.last_timestamp - ulog.start_timestamp)/1e6), 60)
    h2, m2 = divmod(m2, 60)
    print("Logging start time: {:d}:{:02d}:{:02d}, duration: {:d}:{:02d}:{:02d}".format(
        h1, m1, s1, h2, m2, s2))

    dropout_durations = [dropout.duration for dropout in ulog.dropouts]
    if len(dropout_durations) == 0:
        print("No Dropouts")
    else:
        print("Dropouts: count: {:}, total duration: {:.1f} s, max: {:} ms, mean: {:} ms"
              .format(len(dropout_durations), sum(dropout_durations)/1000.,
                      max(dropout_durations),
                      int(sum(dropout_durations)/len(dropout_durations))))

    version = ulog.get_version_info_str()
    if not version is None:
        print('SW Version: {}'.format(version))

    print("Info Messages:")
    for k in sorted(ulog.msg_info_dict):
        if not k.startswith('perf_') or verbose:
            print(" {0}: {1}".format(k, ulog.msg_info_dict[k]))


    if len(ulog.msg_info_multiple_dict) > 0:
        if verbose:
            print("Info Multiple Messages:")
            for k in sorted(ulog.msg_info_multiple_dict):
                print(" {0}: {1}".format(k, ulog.msg_info_multiple_dict[k]))
        else:
            print("Info Multiple Messages: {}".format(
                ", ".join(["[{}: {}]".format(k, len(ulog.msg_info_multiple_dict[k])) for k in
                           sorted(ulog.msg_info_multiple_dict)])))



    print("")
    print("{:<41} {:7}, {:10}".format("Name (multi id, message size in bytes)",
                                      "number of data points", "total bytes"))

    data_list_sorted = sorted(ulog.data_list, key=lambda d: d.name + str(d.multi_id))
    for d in data_list_sorted:
        message_size = sum(ULog.get_field_size(f.type_str) for f in d.field_data)
        num_data_points = len(d.data['timestamp'])
        name_id = "{:} ({:}, {:})".format(d.name, d.multi_id, message_size)
        print(" {:<40} {:7d} {:10d}".format(name_id, num_data_points,
                                            message_size * num_data_points))

    # aaaaaaaaaaaa
    return ulog.msg_info_dict['ver_hw'], ulog.msg_info_dict['sys_name'], version

##################
# 
def main():
    """Commande line interface"""
    parser = argparse.ArgumentParser(description='Display information from an ULog file')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Verbose output', default=False)
    parser.add_argument('-m', '--message', dest='message',
                        help='Show a specific Info Multiple Message')
    parser.add_argument('-n', '--newline', dest='newline', action='store_true',
                        help='Add newline separators (only with --message)', default=False)
    parser.add_argument('-i', '--ignore', dest='ignore', action='store_true',
                        help='Ignore string parsing exceptions', default=False)
    args = parser.parse_args()

    # 
    convert_ulog_info(ulog_file_name=args.filename, message=args.message, newline=args.newline, verbose=args.verbose, disable_str_exceptions=False)

####################################
# 
def convert_ulog_info(ulog_file_name, message, newline=False, verbose=False, disable_str_exceptions=False):

    ulog = ULog(ulog_file_name, None, disable_str_exceptions)
    if message:
        separator = ""
        if newline: separator = "\n"
        if len(ulog.msg_info_multiple_dict) > 0 and message in ulog.msg_info_multiple_dict:
            message_info_multiple = ulog.msg_info_multiple_dict[message]
            for i, m in enumerate(message_info_multiple):
                if len(m) > 0 and isinstance(m[0], (bytes, bytearray)):
                    print("# {} {} (len: {:}):".format(message, i, sum(len(item) for item in m)))
                    print(separator.join(' '.join('{:02x}'.format(x) for x in item) for item in m))
                else:
                    print("# {} {}:".format(message, i))
                    print(separator.join(m))
        else:
            print("message {} not found".format(message))
    else:
        ver_hw, sys_name, version = show_info(ulog, verbose)
        return ver_hw, sys_name, version
    # sdasd
