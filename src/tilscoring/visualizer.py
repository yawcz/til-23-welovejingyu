import argparse
import shelve

from math import ceil
from typing import Dict, Tuple, Any,List,Union
from datetime import datetime

import cv2

from getch import getch, getche
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.style import Style
from .types import *


def valid_style(expr:bool):
    return 'green' if expr else 'red'

highlight_style = Style(bgcolor='yellow')

# sn, id, timestamp, situation, audio, digits , score
def render_report_row(sn:int, report:Report) -> Tuple:
    image_table = Table(show_header=False, box=None)

    if report.id == 'Ended' :
        return  (
            '{:d}'.format(sn),
            Text(report.id),
            Text(report.timestamp.strftime('%H:%M:%S')),
            report.situation,
            None,
            None,
            str(report.score)
        )
    else:
        return (
            '{:d}'.format(sn),
            report.id,
            Text(report.timestamp.strftime('%H:%M:%S'), style=valid_style(report.time_valid)),
            Text(report.situation, style=valid_style(report.time_valid)),
            Text(report.audio, style=valid_style(report.time_valid)),
            Text(str(report.digits), style=valid_style(report.time_valid)),
            Text(str(report.score), style=valid_style(report.time_valid))
        )

def main():
    global active_report_ind

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Shelve filename')
    parser.add_argument('-n', dest='rows_per_page', required=False, default=10, type=int, help='Number of rows per page to display.')

    args = parser.parse_args()

    shelf = shelve.open(args.filename, 'r', protocol=4)

    report_ids = sorted(list(shelf.keys()), reverse=True) # sort reverse chronological

    num_reports = len(report_ids)
    
    table_rows = [render_report_row(i, shelf[rep_id]) for i, rep_id in enumerate(report_ids)]
    num_pages = ceil(len(table_rows) / args.rows_per_page)
    page_num = 0

    active_report_ind = 0
    # print(table_rows)

    def render_table():
        global active_report_ind
       
        grid = Table(box=None)
        
        table1 = Table('S/N', ' ','Report End Time', 'Report Duration', ' ', ' ','Total Score', show_lines=True)
        row1 = table_rows[0]
        table1.add_row(*row1)

        grid.add_row(table1)
        # table_rows.pop(0)

        table = Table('S/N', 'Report ID', 'Report Time', 'Report Situation', 'Report Audio', 'Report Digits','Score', show_lines=True)
        page_rows = table_rows[args.rows_per_page*page_num+1:args.rows_per_page*(page_num+1)]
        
        for row in page_rows:
            table.add_row(*row)

        # highlight active row
        # active_row_ind = active_report_ind % args.rows_per_page
        # table.rows[active_row_ind].style = highlight_style

        grid.add_row(table)
        grid.add_row(Text('Page: {:d}/{:d}'.format(page_num+1, num_pages)))

        return grid 

    # window = cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

    with Live(get_renderable=render_table, refresh_per_second=4, screen=True, auto_refresh=True):
        while True:
           
            # show the image
            report_key = report_ids[active_report_ind]
            report:Report = shelf[report_key]

            # cv2.imshow("Preview", report.get_annotated(3))
            # cv2.waitKey(1)
            direction = ''
            key = getche().lower()
            if key == 'q':
                break
            if key == '\x1b':
                direction = {'[A': 'up', '[B': 'down', '[C': 'right', '[D': 'left'}[getch() + getch()]

            if direction != '':
                if direction == 'up':
                    # decrease index
                    # table.rows[active_row_ind].style = None
                    active_report_ind -= 1
                elif direction == 'down':
                    # increase index
                    # table.rows[active_row_ind].style = None
                    active_report_ind += 1
                elif direction == 'right':
                    page_num += 1
                    page_num = min(page_num, num_pages-1)
                    active_report_ind= page_num*args.rows_per_page
                elif direction == 'left':
                    page_num -= 1
                    page_num = max(page_num, 0)
                    active_report_ind = page_num*args.rows_per_page
            

            # indexing and paging
            active_report_ind = max(0, min(num_reports-1, active_report_ind))
            new_page_num = active_report_ind // args.rows_per_page
            
            page_num = new_page_num
                

if __name__ == '__main__':  
    main()