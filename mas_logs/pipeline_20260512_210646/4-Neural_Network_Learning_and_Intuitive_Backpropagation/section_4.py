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
        # Initialize Scene
        lecture_lines = [
            'Backpropagation calculates the gradient for every internal weight.',
            'Gradients show the direction of the steepest descent.',
            'We flow backward from the output toward the input.',
            'Large arrows indicate weights needing major adjustments.',
            'Tiny arrows point to weights that are nearly correct.'
        ]
        self.setup_layout("Intuitive Backpropagation: Working Backward", lecture_lines)

        # Helper to create nodes
        def create_node():
            return Circle(radius=0.35, color=BLUE, fill_color=BLACK, fill_opacity=1)

        # --- Define Network Components ---
        # Layer 1 (Input)
        n11 = create_node()
        n12 = create_node()
        self.place_at_grid(n11, "B2")
        self.place_at_grid(n12, "D2")
        
        # Layer 2 (Hidden)
        n21 = create_node()
        n22 = create_node()
        self.place_at_grid(n21, "B4")
        self.place_at_grid(n22, "D4")
        
        # Layer 3 (Output)
        n31 = create_node()
        self.place_at_grid(n31, "C6")
        
        nodes = VGroup(n11, n12, n21, n22, n31)

        # Connections (Lines)
        def get_line(c1, c2):
            return Line(c1.get_center(), c2.get_center(), stroke_width=2, color=GRAY)

        w1 = get_line(n11, n21)
        w2 = get_line(n11, n22)
        w3 = get_line(n12, n21)
        w4 = get_line(n12, n22)
        w5 = get_line(n21, n31)
        w6 = get_line(n22, n31)
        weights = VGroup(w1, w2, w3, w4, w5, w6)

        # === Animation for Lecture Line 1 ===
        # 'Backpropagation calculates the gradient for every internal weight.'
        self.lecture[0].set_color(YELLOW)
        self.play(Create(nodes), Create(weights), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'Gradients show the direction of the steepest descent.'
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Display dL/dw symbols next to connections
        lbl_grad1 = Text("dL/dw", font_size=18, color=WHITE)
        lbl_grad2 = Text("dL/dw", font_size=18, color=WHITE)
        self.place_at_grid(lbl_grad1, "A3", scale_factor=0.8) # Near weight w1
        self.place_at_grid(lbl_grad2, "A5", scale_factor=0.8) # Near weight w5 - Adjusted to A5 to avoid overlap
        
        self.play(FadeIn(lbl_grad1), FadeIn(lbl_grad2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'We flow backward from the output toward the input.'
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Gradients (red arrows) flowing backward
        # Layer 3 -> Layer 2
        arr5 = Arrow(n31.get_center(), n21.get_center(), color="#FF0000", buff=0.4, stroke_width=4)
        arr6 = Arrow(n31.get_center(), n22.get_center(), color="#FF0000", buff=0.4, stroke_width=4)
        
        # Layer 2 -> Layer 1
        arr1 = Arrow(n21.get_center(), n11.get_center(), color="#FF0000", buff=0.4, stroke_width=4)
        arr3 = Arrow(n21.get_center(), n12.get_center(), color="#FF0000", buff=0.4, stroke_width=4)
        
        self.play(GrowArrow(arr5), GrowArrow(arr6))
        self.play(GrowArrow(arr1), GrowArrow(arr3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'Large arrows indicate weights needing major adjustments.'
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Flash weight 5 (high gradient) in bright white
        self.play(w5.animate.set_color(WHITE).set_stroke_width(8), run_time=0.4)
        self.play(w5.animate.set_color(GRAY).set_stroke_width(2), run_time=0.4)
        
        # Scale arrow thickness to represent high magnitude
        big_arr5 = Arrow(n31.get_center(), n21.get_center(), color="#FF0000", buff=0.4, stroke_width=15)
        self.play(ReplacementTransform(arr5, big_arr5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'Tiny arrows point to weights that are nearly correct.'
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Scale arrow thickness to represent low magnitude
        tiny_arr6 = Arrow(n31.get_center(), n22.get_center(), color="#FF0000", buff=0.4, stroke_width=1)
        self.play(ReplacementTransform(arr6, tiny_arr6))
        
        # Show internal 'blame' values updating inside nodes
        blame_val1 = Text("0.9", font_size=20, color=WHITE)
        blame_val2 = Text("-0.1", font_size=20, color=WHITE) # Updated value to -0.1 as per issue
        self.place_at_grid(blame_val1, "A4", scale_factor=0.8) # Adjusted to A4 to avoid overlap with n21
        self.place_at_grid(blame_val2, "E4", scale_factor=0.8) # Adjusted to E4 to avoid overlap with n22
        
        self.play(Write(blame_val1), Write(blame_val2))
        self.wait(2)
