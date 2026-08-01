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
        lecture_lines = [
            'Zeros are points where the function equals zero.', 
            'Non-trivial zeros seem to align on one line.', 
            'This critical line sits at real part one half.', 
            'The Riemann Hypothesis predicts they all lie here.', 
            'Proving this unlocks the secrets of prime numbers.'
        ]
        self.setup_layout("The Million Dollar Mystery: The Riemann Hypothesis", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Use YELLOW for the zero concept
        self.lecture[0].set_color(YELLOW)
        
        # Define the Critical Strip area (Re from 0 to 1)
        # Re=0 at Column 2, Re=1 at Column 5
        strip_rect = Rectangle(width=3, height=3, color="#2F4F4F", fill_opacity=0.5, stroke_width=0)
        self.place_in_area(strip_rect, "B2", "E5")
        
        # Boundary lines for the strip
        line_re_0 = Line(self.grid["B2"], self.grid["E2"], color=WHITE, stroke_width=2)
        line_re_1 = Line(self.grid["B5"], self.grid["E5"], color=WHITE, stroke_width=2)
        
        # Labels for the boundaries
        label_0 = Text("0", font_size=20, color=WHITE)
        self.place_at_grid(label_0, "F2")
        label_1 = Text("1", font_size=20, color=WHITE)
        self.place_at_grid(label_1, "F5")
        
        self.play(FadeIn(strip_rect), Create(line_re_0), Create(line_re_1), Write(label_0), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight blue for the critical line
        self.lecture[1].set_color("#1E90FF")
        
        # Calculate position for Re(s) = 1/2 (between col 3 and 4)
        top_cl = (self.grid["B3"] + self.grid["B4"]) / 2
        bottom_cl = (self.grid["E3"] + self.grid["E4"]) / 2
        critical_line = Line(top_cl, bottom_cl, color="#1E90FF", stroke_width=5)
        
        self.play(Create(critical_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Keep lecture white, add specific labels
        self.lecture[2].set_color(WHITE)
        
        cl_label = Text("Critical Line", font_size=20, color="#1E90FF")
        label_pos = (self.grid["A3"] + self.grid["A4"]) / 2
        cl_label.move_to(label_pos)
        
        half_label = Text("1/2", font_size=20, color="#1E90FF")
        half_pos = (self.grid["F3"] + self.grid["F4"]) / 2
        half_label.move_to(half_pos)
        
        self.play(Write(cl_label), Write(half_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Zeros appear on the line
        self.lecture[3].set_color(YELLOW)
        
        zeros = VGroup()
        for i in range(1, 5):
            alpha = i / 5.0
            pos = top_cl * (1 - alpha) + bottom_cl * alpha
            dot = Dot(pos, color=YELLOW, radius=0.08)
            zeros.add(dot)
            
        self.play(LaggedStart(*[FadeIn(z, scale=1.5) for z in zeros], lag_ratio=0.4))
        
        # The outlier dot and rejection
        outlier_pos = self.grid["D4"] # Column 4 is Re=0.66, definitely off-line (Re=0.5)
        outlier_dot = Dot(outlier_pos, color="#FF8C00")
        cross = Cross(outlier_dot, stroke_color=RED, stroke_width=6).scale(0.5)
        
        self.play(FadeIn(outlier_dot))
        self.play(Create(cross))
        self.wait(1)
        self.play(FadeOut(outlier_dot), FadeOut(cross))

        # === Animation for Lecture Line 5 ===
        # Prize money and treasure chest
        self.lecture[4].set_color("#FFD700")
        
        # Create a simple treasure chest representation
        chest_body = Rectangle(width=0.8, height=0.6, color="#FFD700", fill_opacity=0.8)
        chest_lock = Square(side_length=0.15, color=BLACK, fill_opacity=1).move_to(chest_body.get_center())
        chest = VGroup(chest_body, chest_lock)
        # Issue 47 Fix: Move chest to C6 and scale down
        self.place_at_grid(chest, "C6", scale_factor=0.8)
        
        prize_label = Text("$1,000,000", font_size=24, color="#FFD700")
        # Issue 46 Fix: Place prize_label in area D6-E6 and scale down
        self.place_in_area(prize_label, "D6", "E6", scale_factor=0.6)
        
        self.play(FadeIn(chest), Write(prize_label))
        self.wait(3)
