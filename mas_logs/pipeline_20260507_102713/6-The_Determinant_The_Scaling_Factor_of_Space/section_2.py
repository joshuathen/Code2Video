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

class Section2Scene(TeachingScene):
    def construct(self):
        # Section title and lecture lines
        title_str = "Prerequisite: The Unit Square and Basis Vectors"
        lecture_lines = [
            'The unit square is defined by basis vectors.',
            'Matrix columns tell us where these vectors land.',
            'This transformation determines the new shape of space.'
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        I_HAT_COLOR = "#FF0000"
        J_HAT_COLOR = "#0000FF"
        SQUARE_COLOR = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Display unit vectors i-hat and j-hat forming a unit square
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create a local coordinate system
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, "B2", "F6", scale_factor=0.6)
        
        # Basis Vectors
        i_hat = Vector(plane.c2p(1, 0) - plane.c2p(0, 0), color=I_HAT_COLOR).move_to(plane.c2p(0.5, 0), aligned_edge=ORIGIN)
        j_hat = Vector(plane.c2p(0, 1) - plane.c2p(0, 0), color=J_HAT_COLOR).move_to(plane.c2p(0, 0.5), aligned_edge=ORIGIN)
        
        # Labels
        i_label = Text("i", slant=ITALIC, color=I_HAT_COLOR, font_size=24)
        j_label = Text("j", slant=ITALIC, color=J_HAT_COLOR, font_size=24)
        i_label.next_to(i_hat, DOWN, buff=0.1)
        j_label.next_to(j_hat, LEFT, buff=0.1)

        # Asset: Unit Square
        unit_square_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/square.svg")
        unit_square_asset.set_stroke(SQUARE_COLOR, width=2).set_fill(SQUARE_COLOR, opacity=0.3)
        square_width = plane
