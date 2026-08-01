from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup context with updated script from prompt
        title = "Real-World Application: Network Topology"
        lines = [
            'Subway systems often start as messy geographic maps.',
            'Topology transforms them into clean, readable schematics.',
            'We focus only on paths connecting specific stations.',
            "Distorting the map doesn't break these vital connections.",
            'In this network, connectivity is the only invariant.'
        ]
        self.setup_layout(title, lines)
        
        # Define colors and assets
        GRAY_MAP = "#999999"
        BLUE_PATH = "#0072B2"
        TEXT_WHITE = "#FFFFFF"
        SUBWAY_ICON_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/subway.svg"

        # === Animation for Lecture Line 1 ===
        # Subway systems often start as messy geographic maps.
        self.lecture[0].set_color(BLUE_PATH)
        
        # Grid positions for nodes selected to facilitate schematic alignment in phase 2
        pos_keys = ["B3", "C5", "D2", "E3", "C3"]
        pos_vals = [self.grid[k] for k in pos_keys]

        # Use subway icon for transit nodes [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/subway.svg]
        nodes_geo = VGroup()
        for pos in pos_vals:
            # SVGMobject handles internal paths; set_color applies to entire icon
            icon = SVGMobject(SUBWAY_ICON_PATH, color=GRAY_MAP).scale(0.15).move_to(pos)
            nodes_geo.add(icon)

        # Messy curved streets (using fixed offsets for deterministic "geographic" look)
        def get_messy_curve(p1, p2, offset):
            mid = (p1 + p2) / 2 + offset
            return CubicBezier(p1, mid + UP*0.3, mid + DOWN*0.3, p2, color=GRAY_MAP)

        offsets = [
            np.array([0.2, 0.3, 0]),  # Connection N1-N5
            np.array([-0.3, 0.1, 0]), # Connection N5-N2
            np.array([0.1, -0.2, 0]), # Connection N5-N3
            np.array([-0.2, -0.3, 0]),# Connection N3-N4
            np.array([0.3, -0.1, 0])  # Connection N2-N4
        ]

        edges_geo = VGroup(
            get_messy_curve(pos_vals[0], pos_vals[4], offsets[0]),
            get_messy_curve(pos_vals[4], pos_vals[1], offsets[1]),
            get_messy_curve(pos_vals[4], pos_vals[2], offsets[2]),
            get_messy_curve(pos_vals[2], pos_vals[3], offsets[3]),
            get_messy_curve(pos_vals[1], pos_vals[3], offsets[4])
        )

        self.play(Create(edges_geo), Create(nodes_geo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Topology transforms them into clean, readable schematics.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_PATH)

        # Clean schematic edges (straight lines). Selected grid points result in 90/45 degree angles.
        edges_schematic = VGroup(
            Line(pos_vals[0], pos_vals[4], color=GRAY_MAP), # Vertical
            Line(pos_vals[4], pos_vals[1], color=GRAY_MAP), # Horizontal
            Line(pos_vals[4], pos_vals[2], color=GRAY_MAP), # 45 deg
            Line(pos_vals[2], pos_vals[3], color=GRAY_MAP), # 45 deg
            Line(pos_vals[1], pos_vals[3], color=GRAY_MAP)  # 45 deg
        )

        self.play(ReplacementTransform(edges_geo, edges_schematic))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We focus only on paths connecting specific stations.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE_PATH)

        # Highlight a specific path between two nodes in bright blue (#0072B2)
        # Path: N1 -> N5 -> N2
        path_segments = VGroup(
            Line(pos_vals[0], pos_vals[4], color=BLUE_PATH, stroke_width=6),
            Line(pos_vals[4], pos_vals[1], color=BLUE_PATH, stroke_width=6)
        )
        
        self.play(Create(path_segments), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Distorting the map doesn't break these vital connections.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE_PATH)

        # Define destination points on the grid for distortion
        dist_keys = ["A4", "B6", "E1", "F5", "D4"]
        dist_vals = [self.grid[k] for k in dist_keys]

        # Use updaters to maintain connectivity during the movement of icons
        def update_edges(mob):
            # Edges: N1-N5, N5-N2, N5-N3, N3-N4, N2-N4
            mob[0].put_start_and_end_on(nodes_geo[0].get_center(), nodes_geo[4].get_center())
            mob[1].put_start_and_end_on(nodes_geo[4].get_center(), nodes_geo[1].get_center())
            mob[2].put_start_and_end_on(nodes_geo[4].get_center(), nodes_geo[2].get_center())
            mob[3].put_start_and_end_on(nodes_geo[2].get_center(), nodes_geo[3].get_center())
            mob[4].put_start_and_end_on(nodes_geo[1].get_center(), nodes_geo[3].get_center())

        def update_path(mob):
            # Path segments stay anchored to N1-N5 and N5-N2
            mob[0].put_start_and_end_on(nodes_geo[0].get_center(), nodes_geo[4].get_center())
            mob[1].put_start_and_end_on(nodes_geo[4].get_center(), nodes_geo[1].get_center())

        edges_schematic.add_updater(update_edges)
        path_segments.add_updater(update_path)

        self.play(
            nodes_geo[0].animate.move_to(dist_vals[0]),
            nodes_geo[1].animate.move_to(dist_vals[1]),
            nodes_geo[2].animate.move_to(dist_vals[2]),
            nodes_geo[3].animate.move_to(dist_vals[3]),
            nodes_geo[4].animate.move_to(dist_vals[4]),
            run_time=3,
            rate_func=smooth
        )
        
        edges_schematic.remove_updater(update_edges)
        path_segments.remove_updater(update_path)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # In this network, connectivity is the only invariant.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE_PATH)

        # Final invariant text
        invariant_text = Text("CONNECTIVITY IS INVARIANT", font_size=24, color=TEXT_WHITE)
        # Position label at bottom-center of the visual area using place_in_area (Fix for Issue 42)
        self.place_in_area(invariant_text, 'F1', 'F6', scale_factor=0.7)

        self.play(Write(invariant_text))
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
