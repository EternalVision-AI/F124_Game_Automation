import d3dshot

d = d3dshot.create()
d.screenshot_to_disk(directory=".", file_name="test.png")