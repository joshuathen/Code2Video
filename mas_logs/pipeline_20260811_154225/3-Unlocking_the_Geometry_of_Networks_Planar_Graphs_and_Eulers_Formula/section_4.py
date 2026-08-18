from manim import *

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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dual vertices equal original faces.",
            "Dual faces equal original vertices.",
            "Duality preserves the Euler balance.",
            "Simple squares demonstrate this mapping.",
            "Duality swaps these network properties."
        ]
        self.setup_layout("Deep Dive: Why Duality Matters", lecture_lines)
        
        # Asset path
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        
        # Create square graph (V=4, E=4, F=2)
        # Using a square representation
        orig_vertices = {0: UP+LEFT, 1: UP+RIGHT, 2: DOWN+RIGHT, 3: DOWN+LEFT}
        orig_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        original = Graph(list(orig_vertices.keys()), orig_edges, layout=orig_vertices, 
                         vertex_config={"radius": 0.1, "color": "#33FF57"}, edge_config={"stroke_width": 4, "color": "#33FF57"})
        
        # Label square
        sq_icon = SVGMobject(asset_path).set_color("#33FF57")
        orig_group = VGroup(original, sq_icon).arrange(DOWN)
        self.place_in_area(orig_group, 'C4', 'E6', scale_factor=0.6)
        
        # Dual graph (V=2, E=4, F=4) - a diamond shape
        dual_nodes = {0: UP, 1: DOWN}
        dual_edges = [(0, 1)] # Simple representation
        dual = Graph(list(dual_nodes.keys()), dual_edges, layout=dual_nodes, vertex_config={"radius": 0.1, "color": TEAL})
        
        dual_group = VGroup(dual, SVGMobject(asset_path).set_color(TEAL)).arrange(DOWN)
        self.place_in_area(dual_group, 'A4', 'B6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(orig_group))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(dual_group))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(orig_group), Indicate(dual_group), run_time=1.5)
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        self.play(FadeOut(orig_group), FadeOut(dual_group))
        self.lecture[3].set_color(TEAL)
        self.place_at_grid(orig_group, 'C4', scale_factor=0.7)
        self.play(FadeIn(orig_group))

        # === Animation for Lecture Line 5 ===
        self.play(Rotate(orig_group, angle=PI), run_time=2)
        self.lecture[4].set_color(TEAL)
