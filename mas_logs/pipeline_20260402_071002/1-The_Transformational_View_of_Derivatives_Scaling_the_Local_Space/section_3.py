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
        # Correct Lecture Lines from prompt
        lecture_lines_text = [
            "The derivative acts as a local multiplication constant.",
            "At input x equals three, the scaling is six.",
            "We define a tiny input change as dx.",
            "This produces a corresponding output change called df.",
            "Here, df is exactly six times larger than dx."
        ]
        self.setup_layout("The Math: df = f'(x) * dx", lecture_lines_text)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line in green to match the formula
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Display the formula 'df = f'(x) dx' using green (#00FF00)
        formula = Text("df = f'(x) dx", color="#00FF00", font_size=32)
        # Fix Issue 29: Use optimal placement and scale
        self.place_in_area(formula, 'A3', 'B5', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line in white
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # Show calculation context in white (#FFFFFF)
        context_text = Text("x = 3,  f'(3) = 6", color="#FFFFFF", font_size=24)
        # Fix Issue 30: Positioning context text properly
        self.place_in_area(context_text, 'C3', 'C5', scale_factor=0.8)
        self.play(Write(context_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line in yellow to match dx
        self.play(self.lecture[2].animate.set_color("#FFFF00"))

        # Setup Input line using row E
        input_line = Line(self.grid['E1'], self.grid['E6'], color=GRAY)
        input_label = Text("Input", font_size=16, color=GRAY).next_to(input_line, LEFT, buff=0.2)
        
        # Define the small yellow segment dx on the Input line
        dx_len = 0.2
        dx_start_pos = self.grid['E2']
        dx_seg = Line(dx_start_pos, dx_start_pos + RIGHT * dx_len, color="#FFFF00", stroke_width=8)
        dx_label = Text("dx", color="#FFFF00", font_size=18).next_to(dx_seg, UP, buff=0.1)

        self.play(Create(input_line), Write(input_label))
        self.play(Create(dx_seg), Write(dx_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight fourth line in magenta to match df
        self.play(self.lecture[3].animate.set_color("#FF00FF"))

        # Setup Output line using row F
        output_line = Line(self.grid['F1'], self.grid['F6'], color=GRAY)
        output_label = Text("Output", font_size=16, color=GRAY).next_to(output_line, LEFT, buff=0.2)
        
        # Define the magenta segment df on the Output line
        # Scale is 6, so df_len = 0.2 * 6 = 1.2
        df_len = dx_len * 6
        df_start_pos = self.grid['F2']
        df_seg = Line(df_start_pos, df_start_pos + RIGHT * df_len, color="#FF00FF", stroke_width=8)
        df_label = Text("df", color="#FF00FF", font_size=18).next_to(df_seg, DOWN, buff=0.1)

        self.play(Create(output_line), Write(output_label))
        self.play(Create(df_seg), Write(df_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight fifth line in white
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Animate the dx segment being 'transformed' and scaled 6 times to match df
        dx_ghost = dx_seg.copy()
        
        # Move ghost to start position of df_seg
        target_pos = df_seg.get_start() + RIGHT * (dx_len / 2)
        self.play(dx_ghost.animate.move_to(target_pos), run_time=1)
        
        # Scale the ghost by factor 6 relative to its start point
        self.play(dx_ghost.animate.scale(6, about_point=df_seg.get_start()), run_time=1.5)
        
        # Conclude by flashing the final scaled segment
        self.play(FadeOut(dx_ghost))
        self.play(Indicate(df_seg, color="#FF00FF"))
        
        self.wait(2)
