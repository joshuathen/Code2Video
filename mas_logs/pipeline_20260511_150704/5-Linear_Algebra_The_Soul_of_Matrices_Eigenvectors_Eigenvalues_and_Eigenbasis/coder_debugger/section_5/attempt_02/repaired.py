from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets)
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
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
        # Setup the title and lecture lines
        lecture_lines = [
            'An eigenbasis uses eigenvectors as coordinate axes.',
            'In this view, the transformation is purely diagonal.',
            'Complex operations become simple scalar multiplications.',
            'It makes calculating matrix powers incredibly efficient.',
            'Changing perspective simplifies the entire transformation.'
        ]
        self.setup_layout("The Power of Eigenbasis", lecture_lines)

        # Define basic colors
        EIGENVEC1_COLOR = "#00FFFF"
        EIGENVEC2_COLOR = "#FF00FF"
        DIAGONAL_COLOR = "#FFFF00"
        HIGHLIGHT_COLOR = "#FFFF00"

        # Define vectors
        v1_coords = np.array([1.5, 0.75, 0])
        v2_coords = np.array([-0.75, 1.5, 0])
        
        # Create NumberPlane and group for the visual components
        plane = NumberPlane(
            x_range=[-3, 3, 1], 
            y_range=[-3, 3, 1], 
            x_length=4.5, 
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        # Eigenvectors - Using Text to avoid LaTeX dependency (fixes FileNotFoundError: 'latex')
        v1 = Vector(v1_coords, color=EIGENVEC1_COLOR)
        v2 = Vector(v2_coords, color=EIGENVEC2_COLOR)
        
        # Use Text instead of MathTex to bypass missing latex installation
        v1_label = Text("v1", color=EIGENVEC1_COLOR).scale(0.6)
        v2_label = Text("v2", color=EIGENVEC2_COLOR).scale(0.6)
        
        v1_label.next_to(v1.get_end(), UR, buff=0.1)
        v2_label.next_to(v2.get_end(), UL, buff=0.1)
        
        visual_group = VGroup(plane, v1, v2, v1_label, v2_label)
        self.place_in_area(visual_group, "B2", "E5")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(v1), Write(v1_label), GrowArrow(v2), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Rotation calculation to align v1 with x-axis
        angle = -v1.get_angle()
        
        self.play(
            Rotate(visual_group, angle=angle, about_point=visual_group.get_center()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Matrix D display - Using Text for stability
        matrix_d = Text("D = [[L1, 0], [0, L2]]", color=DIAGONAL_COLOR, font_size=24)
        self.place_at_grid(matrix_d, "A4", scale_factor=0.9)
        
        self.play(Write(matrix_d))
        self.play(
            v1.animate.scale(1.5, about_point=visual_group.get_center()),
            v2.animate.scale(0.5, about_point=visual_group.get_center()),
            v1_label.animate.shift(v1.get_vector() * 0.2),
            v2_label.animate.shift(v2.get_vector() * -0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Efficiency Formula
        formula = Text("A^n = P D^n P^-1", color=WHITE, font_size=24)
        self.place_at_grid(formula, "F4", scale_factor=1.1)
        
        self.play(Write(formula))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Emphasize simplicity - highlight D^n
        dn_highlight = Text("D^n = [[L1^n, 0], [0, L2^n]]", color=DIAGONAL_COLOR, font_size=22)
        self.place_at_grid(dn_highlight, "C6", scale_factor=0.8)
        
        self.play(Write(dn_highlight))
        self.play(Indicate(dn_highlight))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
        self.wait(1)