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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup colors
        COLOR_A = YELLOW
        COLOR_B = "#58C4DD" # BLUE_B
        COLOR_V = "#83C167" # GREEN_C
        
        self.setup_layout("The Right-to-Left Rule", [
            'In notation, the first action sits on the right.', 
            'Like a conveyor belt, vectors enter from the right.', 
            'Matrix A acts first, then Matrix B follows.'
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_A))
        
        # Formula: (BA)v - Replaced MathTex with Text to avoid LaTeX dependency
        formula = Text("(BA)v", font_size=48)
        formula[1].set_color(COLOR_B) # B
        formula[2].set_color(COLOR_A) # A
        formula[4].set_color(COLOR_V) # v
        # Fix for Issue #36: Moved from B2-B5 to C2-C5 for better grid utilization
        self.place_in_area(formula, 'C2', 'C5', scale_factor=1.1)
        
        self.play(Write(formula))
        self.wait(0.5)
        
        # Sequential highlighting from right to left: v -> A -> B
        self.play(Indicate(formula[4], color=COLOR_V))
        self.play(Indicate(formula[2], color=COLOR_A))
        self.play(Indicate(formula[1], color=COLOR_B))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_B))
        
        # Conveyor belt visuals
        belt = Line(self.grid['D6'], self.grid['D1'], color=GRAY_A, stroke_width=2)
        
        machine_a_box = Square(side_length=0.8, color=COLOR_A, fill_opacity=0.2)
        label_a = Text("A", color=COLOR_A, font_size=24).move_to(machine_a_box)
        machine_a = VGroup(machine_a_box, label_a)
        self.place_at_grid(machine_a, 'D4')
        
        machine_b_box = Square(side_length=0.8, color=COLOR_B, fill_opacity=0.2)
        label_b = Text("B", color=COLOR_B, font_size=24).move_to(machine_b_box)
        machine_b = VGroup(machine_b_box, label_b)
        self.place_at_grid(machine_b, 'D2')
        
        # Replaced MathTex with Text to avoid LaTeX dependency
        vector_v = Text("v", color=COLOR_V, font_size=36, slant=ITALIC)
        self.place_at_grid(vector_v, 'D6')
        
        self.play(Create(belt))
        self.play(FadeIn(machine_a), FadeIn(machine_b))
        self.play(FadeIn(vector_v))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_V))
        
        # Vector v enters machine A then machine B
        self.play(vector_v.animate.move_to(self.grid['D4']), run_time=1.5)
        self.play(Indicate(machine_a, color=WHITE))
        self.play(vector_v.animate.move_to(self.grid['D2']), run_time=1.5)
        self.play(Indicate(machine_b, color=WHITE))
        self.play(vector_v.animate.move_to(self.grid['D1']), run_time=1)
        
        # Right-to-Left Label and Arrows
        arrow_flow = Arrow(start=self.grid['E5'], end=self.grid['E2'], color=COLOR_V, buff=0.1)
        label_flow = Text("Right-to-Left Order", color=COLOR_V, font_size=20)
        # Fix for Issue #35: Moved from point F3 to area E2-E4 for better alignment and to prevent clipping
        self.place_in_area(label_flow, 'E2', 'E4', scale_factor=0.8)
        
        self.play(GrowArrow(arrow_flow), Write(label_flow))
        self.wait(2)
