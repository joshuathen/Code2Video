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

class Section2Scene(TeachingScene):
    def construct(self):
        # Fetch data from storyboard and outline
        title_text = "Prerequisite: The Concept of Slope"
        lecture_lines = [
            "Slope measures how vertical change relates to horizontal change.",
            "For straight lines, this rate of change stays constant.",
            "We calculate it using the simple 'Rise over Run'."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors based on requirements
        COLOR_LINE = "#00FF00"      # Green
        COLOR_HIKER = "#FFFFFF"     # White
        COLOR_TRIANGLE = "#00FFFF"  # Cyan
        COLOR_FORMULA = "#FFFF00"   # Yellow

        # === Animation for Lecture Line 1 ===
        # L1: Slope measures how vertical change relates to horizontal change.
        self.play(self.lecture[0].animate.set_color(COLOR_LINE))
        
        # Define line from E2 to B5 (positive slope)
        start_pt = self.grid["E2"]
        end_pt = self.grid["B5"]
        slope_line = Line(start_pt, end_pt, color=COLOR_LINE, stroke_width=4)
        
        # Hiker representation: Using a triangle as a surrogate for hiker icon
        hiker = Triangle(color=COLOR_HIKER, fill_opacity=1).scale(0.15)
        hiker.move_to(start_pt + UP * 0.25)
        
        self.play(Create(slope_line), FadeIn(hiker))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L2: For straight lines, this rate of change stays constant.
        self.play(self.lecture[1].animate.set_color(COLOR_TRIANGLE))
        
        # Right-angled triangle under the line
        # Vertices: Start(E2), Corner(E5), End(B5)
        corner_pt = self.grid["E5"]
        triangle_poly = Polygon(start_pt, corner_pt, end_pt, color=COLOR_TRIANGLE, stroke_width=2)
        
        # Labels for the triangle sides
        # "Rise" for the vertical side (E5 to B5)
        rise_label = Text("Rise", font_size=24, color=COLOR_TRIANGLE)
        # Resolved Issue 32: Position at D6
        self.place_at_grid(rise_label, "D6", scale_factor=0.8)
        
        # "Run" for the horizontal side (E2 to E5)
        run_label = Text("Run", font_size=24, color=COLOR_TRIANGLE)
        # Resolved Issue 33: Position at F3
        self.place_at_grid(run_label, "F3", scale_factor=0.8)
        
        self.play(Create(triangle_poly), Write(rise_label), Write(run_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L3: We calculate it using the simple 'Rise over Run'.
        self.play(self.lecture[2].animate.set_color(COLOR_FORMULA))
        
        # Formula: Slope = Rise / Run
        # Resolved Issue 31: Positioning in the A1-B3 area
        formula = MathTex(r"\text{Slope} = \frac{\text{Rise}}{\text{Run}}", color=COLOR_FORMULA)
        self.place_in_area(formula, "A1", "B3", scale_factor=0.8)
        
        # Midpoint of the line for hiker movement
        midpoint = (start_pt + end_pt) / 2
        
        self.play(
            hiker.animate.move_to(midpoint + UP * 0.25),
            Write(formula)
        )
        self.wait(3)
