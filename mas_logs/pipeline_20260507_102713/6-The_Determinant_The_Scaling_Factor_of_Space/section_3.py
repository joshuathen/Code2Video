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
        # Configuration
        TITLE = "Geometric Definition: Area of the Parallelogram"
        LINES = [
            'Transformed basis vectors form a skewed parallelogram.', 
            "The determinant measures this parallelogram's area.", 
            'For a 2x2 matrix, use the formula ad minus bc.', 
            'This number quantifies how much space has stretched.', 
            'An area of five means a five-fold expansion.'
        ]
        
        self.setup_layout(TITLE, LINES)
        
        # Colors
        COLOR_PARALLELOGRAM = "#E6E6FA" # Light Purple
        COLOR_FORMULA = "#FFFFFF"      # White
        COLOR_RESULT = "#FFFF00"       # Yellow
        COLOR_V1 = RED
        COLOR_V2 = BLUE
        
        # Coordinate system setup
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 5, 1],
            x_length=3.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.2, "stroke_color": GREY}
        )
        self.place_in_area(plane, "B2", "F6", scale_factor=0.9)
        
        origin = plane.c2p(0, 0)
        
        # Unit square mobjects
        unit_square = Polygon(
            plane.c2p(0,0), plane.c2p(1,0), plane.c2p(1,1), plane.c2p(0,1),
            stroke_width=2, stroke_color=WHITE, fill_opacity=0.3, fill_color=WHITE
        )
        
        # Transformed basis vectors for matrix [[2, 1], [1, 3]]
        v1_end = plane.c2p(2, 1)
        v2_end = plane.c2p(1, 3)
        v1_vec = Arrow(origin, v1_end, buff=0, color=COLOR_V1, stroke_width=4)
        v2_vec = Arrow(origin, v2_end, buff=0, color=COLOR_V2, stroke_width=4)
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        v1_label = Text("î'", color=COLOR_V1, font_size=24, slant=ITALIC).next_to(v1_end, DR, buff=0.1)
        v2_label = Text("ĵ'", color=COLOR_V2, font_size=24, slant=ITALIC).next_to(v2_end, UL, buff=0.1)

        # Transformed parallelogram
        parallelogram = Polygon(
            plane.c2p(0,0), plane.c2p(2,1), plane.c2p(3,4), plane.c2p(1,3),
            stroke_width=4, stroke_color=COLOR_PARALLELOGRAM,
            fill_color=COLOR_PARALLELOGRAM, fill_opacity=0.5
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.add(plane)
        self.play(Create(unit_square))
        self.wait(0.5)
        self.play(
            ReplacementTransform(unit_square, parallelogram),
            GrowArrow(v1_vec),
            GrowArrow(v2_vec),
            Write(v1_label),
            Write(v2_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        area_text = Text("Area", font_size=28, color=COLOR_PARALLELOGRAM)
        area_text.move_to(plane.c2p(1.5, 2))
        
        self.play(Write(area_text))
        self.play(Indicate(parallelogram, color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        formula = Text("det(A) = (2 × 3) - (1 × 1)", color=COLOR_FORMULA, font_size=28)
        self.place_in_area(formula, "A2", "A5")
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
