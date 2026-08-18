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

class Section1Scene(TeachingScene):
    def construct(self):
        title_color = "#ADD8E6"
        line3_color = "#FFFACD"
        line4_color = "#90EE90"
        line5_color = "#FFD700"

        lecture_lines = [
            "Traditional derivatives focus on slopes of tangent lines.",
            "Let's explore a more transformational perspective instead.",
            "Imagine two parallel number lines: Input and Output.",
            "Function f maps points from top to bottom line.",
            "Watch how f(2) moves from top to bottom."
        ]
        self.setup_layout("Beyond the Slope: The Two-Line Model", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(title_color)
        self.title.set_color(title_color)
        
        # Display title 'Beyond the Slope' and two horizontal lines
        line_in_initial = Line(start=self.grid['B1'], end=self.grid['B6'], color=WHITE)
        line_out_initial = Line(start=self.grid['E1'], end=self.grid['E6'], color=WHITE)
        
        self.play(Create(line_in_initial), Create(line_out_initial), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(line3_color)
        
        # Label the top line 'Input Space' and bottom line 'Output Space'
        label_in = Text("Input Space", font_size=20, color=line3_color)
        label_out = Text("Output Space", font_size=20, color=line3_color)
        
        # FIX: Using place_in_area as requested by critic
        self.place_in_area(label_in, 'A1', 'A2', scale_factor=0.8)
        self.place_in_area(label_out, 'D1', 'D2', scale_factor=0.8)
        
        # Upgrade to NumberLines for measurement
        new_line_in = NumberLine(x_range=[0, 5, 1], length=5, include_numbers=True, font_size=16)
        new_line_out = NumberLine(x_range=[0, 5, 1], length=5, include_numbers=True, font_size=16)
        self.place_in_area(new_line_in, 'B1', 'B6')
        self.place_in_area(new_line_out, 'E1', 'E6')
        
        self.play(
            Write(label_in), 
            Write(label_out),
            ReplacementTransform(line_in_initial, new_line_in),
            ReplacementTransform(line_out_initial, new_line_out),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(line4_color)
        
        # Place a point at x=2 on the top line and f(x)=4 on the bottom
        # x=2 maps to column 3 (B3), x=4 maps to column 5 (E5)
        p_in = Dot(color=line4_color)
        self.place_at_grid(p_in, 'B3')
        
        p_out = Dot(color=line4_color)
        self.place_at_grid(p_out, 'E5')
        
        label_f2 = MathTex("x=2", font_size=24, color=line4_color)
        label_f4 = MathTex("f(2)=4", font_size=24, color=line4_color)
        
        # FIX: Using place_in_area as requested by critic
        self.place_in_area(label_f2, 'A3', 'A3', scale_factor=0.8)
        self.place_in_area(label_f4, 'F5', 'F5', scale_factor=0.8)
        
        self.play(FadeIn(p_in), FadeIn(p_out), Write(label_f2), Write(label_f4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(line5_color)
        
        # Animate an arrow mapping x=2 to f(x)=4
        mapping_arrow = Arrow(start=p_in.get_center(), end=p_out.get_center(), buff=0.1, color=line5_color)
        
        self.play(GrowArrow(mapping_arrow))
        self.play(Indicate(mapping_arrow, color=line5_color))
        self.wait(2)
