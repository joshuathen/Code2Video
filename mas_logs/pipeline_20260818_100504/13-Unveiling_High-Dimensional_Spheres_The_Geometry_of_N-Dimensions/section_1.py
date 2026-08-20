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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We build dimensions from 1D, to 2D, to 3D.",
            "The algebraic form grows with each new dimension.",
            "An n-sphere exists in n+1 dimensional space."
        ]
        self.setup_layout("Prerequisite Review: Building Dimensions", lecture_lines)
        
        # Define assets
        dot_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg").set_color("#FFFFFF")
        line_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg").set_color("#FF5733")
        circle_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg").set_color("#33FF57")
        
        # Labels
        label_dot = Text("0D Point", font_size=20, color="#FFFFFF")
        label_line = Text("1D Segment", font_size=20, color="#FF5733")
        label_circle = Text("2D Manifold", font_size=20, color="#33FF57")
        
        # === Animation for Lecture Line 1 ===
        # Create a dot representing a 0D point (#FFFFFF)
        self.place_at_grid(dot_asset, "B2", scale_factor=0.5)
        label_dot.next_to(dot_asset, DOWN)
        self.play(FadeIn(dot_asset), Write(label_dot))
        
        # === Animation for Lecture Line 2 ===
        # Create a line representing a 1D segment (#FF5733)
        self.place_at_grid(line_asset, "B4", scale_factor=0.5)
        label_line.next_to(line_asset, DOWN)
        self.play(FadeIn(line_asset), Write(label_line))
        self.lecture[0].set_color("#FF5733")
        
        # === Animation for Lecture Line 3 ===
        # Create a circle representing a 2D manifold (#33FF57)
        self.place_at_grid(circle_asset, "B6", scale_factor=0.5)
        label_circle.next_to(circle_asset, DOWN)
        self.play(FadeIn(circle_asset), Write(label_circle))
        self.lecture[1].set_color("#33FF57")
        
        self.wait(2)
