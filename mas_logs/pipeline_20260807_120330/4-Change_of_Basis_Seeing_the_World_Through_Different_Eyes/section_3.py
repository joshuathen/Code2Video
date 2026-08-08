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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title = "Visualizing a New Basis"
        lecture_lines = [
            "- Let's define a new basis using vectors b1 and b2.",
            "- These vectors form a squashed and rotated grid.",
            "- The space remains fixed while our ruler changes."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Define colors from the prompt/storyboard
        ORANGE = "#FFA500"
        CYAN = "#00FFFF"
        GREY = "#696969"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE)
        
        # 1. Dim grey (#696969) standard grid background.
        # Adjusted position and scale per Issue 30
        std_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GREY, "stroke_opacity": 0.5},
            axis_config={"stroke_color": GREY}
        )
        self.place_in_area(std_grid, 'A3', 'F6', scale_factor=0.5)
        origin = std_grid.get_origin()
        
        self.play(Create(std_grid))
        
        # 2. Basis vectors b1 (orange #FFA500), b2 (cyan #00FFFF) from origin.
        # Use coordinates relative to the standard grid's unit size
        b1_target = std_grid.coords_to_point(1.5, 0.5)
        b2_target = std_grid.coords_to_point(0.5, 1.5)
        
        b1_arrow = Arrow(origin, b1_target, buff=0, color=ORANGE)
        b2_arrow = Arrow(origin, b2_target, buff=0, color=CYAN)
        
        b1_label = MathTex(r"\vec{b}_1", color=ORANGE).scale(0.8)
        b2_label = MathTex(r"\vec{b}_2", color=CYAN).scale(0.8)
        
        # Position labels near the vector heads
        b1_label.next_to(b1_arrow.get_end(), RIGHT + DOWN, buff=0.1)
        b2_label.next_to(b2_arrow.get_end(), LEFT + UP, buff=0.1)
        
        self.play(GrowArrow(b1_arrow), Write(b1_label))
        self.play(GrowArrow(b2_arrow), Write(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(CYAN)
        
        # 3. New skewed grid using orange and cyan lines.
        # Basis matrix
        matrix = [[1.5, 0.5], [0.5, 1.5]]
        
        skewed_grid = VGroup()
        # Create a set of grid lines that will be transformed
        # Vertical lines (transformed unit x steps)
        for x in range(-3, 4):
            l = Line(np.array([x, -3, 0]), np.array([x, 3, 0]), color=ORANGE, stroke_opacity=0.6, stroke_width=2)
            skewed_grid.add(l)
        # Horizontal lines (transformed unit y steps)
        for y in range(-3, 4):
            l = Line(np.array([-3, y, 0]), np.array([3, y, 0]), color=CYAN, stroke_opacity=0.6, stroke_width=2)
            skewed_grid.add(l)
            
        # Apply the basis transformation
        skewed_grid.apply_matrix(matrix)
        
        # Adjusted position and scale per Issue 31
        self.place_in_area(skewed_grid, 'A3', 'F6', scale_factor=0.5)
        
        self.play(Create(skewed_grid))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Show a fixed vector v = [2, 2] in standard coordinates
        v_target = std_grid.coords_to_point(2, 2)
        v_arrow = Arrow(origin, v_target, buff=0, color=WHITE, stroke_width=6)
        v_label = MathTex(r"\vec{v}", color=WHITE).scale(0.9)
        v_label.next_to(v_arrow.get_end(), UP + RIGHT, buff=0.1)
        
        self.play(Create(v_arrow), Write(v_label))
        self.play(Indicate(v_arrow, color=WHITE))
        self.wait(2)
