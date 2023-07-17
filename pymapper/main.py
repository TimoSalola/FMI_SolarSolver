import mapper

def plot_map():
    mapper.toggle_grid(True)
    mapper.create_map()
    mapper.add__road_details()
    mapper.plot_point(61, 25)
    mapper.limit_map_to_region()
    mapper.add_text_to_map("kissoja täällä", 60, 25)
    mapper.plot_address("kauppakatu 10")
    mapper.add_title("Test map")
    mapper.show_map()


plot_map()