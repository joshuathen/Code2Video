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
        # Setup title and lecture lines
        title_text = "Prerequisites: Signals and Kernels"
        lecture_lines = [
            "Discrete signals are simple sequences of numeric values.",
            "Kernels are small weight arrays that define transformations.",
            "Indices n and k track positions within these sequences."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors from Storyboard
        BLUE_SIGNAL = "#0000FF"
        RED_KERNEL = "#FF0000"
        GREEN_HIGHLIGHT = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # "Discrete signals are simple sequences of numeric values."
        self.play(self.lecture[0].animate.set_color(BLUE_SIGNAL))
        
        # Signal visualization: [1, 2, 3, 2, 1]
        signal_values = [1, 2, 3, 2, 1]
        signal_bars = VGroup(*[
            Rectangle(
                height=v * 0.4, 
                width=0.4, 
                fill_color=BLUE_SIGNAL, 
                fill_opacity=0.7, 
                stroke_color=WHITE, 
                stroke_width=1
            ) for v in signal_values
        ]).arrange(RIGHT, buff=0.1)
        
        # Place signal in area B2 to C6 (Issue 24 fix)
        self.place_in_area(signal_bars, "B2", "C6")
        
        signal_label = MathTex("f[n]", color=BLUE_SIGNAL)
        # Place signal label at A4 (Issue 24 fix)
        self.place_at_grid(signal_label, "A4")
        
        self.play(Create(signal_bars), Write(signal_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Kernels are small weight arrays that define transformations."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED_KERNEL)
        )
        
        # Kernel visualization: [0.5, 0.5]
        kernel_values = [0.5, 0.5]
        kernel_bars = VGroup(*[
            Rectangle(
                height=v * 0.8, 
                width=0.4, 
                fill_color=RED_KERNEL, 
                fill_opacity=0.7, 
                stroke_color=WHITE, 
                stroke_width=1
            ) for v in kernel_values
        ]).arrange(RIGHT, buff=0.1)
        
        # Place kernel in area E2 to E3
        self.place_in_area(kernel_bars, "E2", "E3")
        
        kernel_label = MathTex("g[k]", color=RED_KERNEL)
        self.place_at_grid(kernel_label, "D2")
        
        self.play(Create(kernel_bars), Write(kernel_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Indices n and k track positions within these sequences."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN_HIGHLIGHT)
        )
        
        # Highlight index 'n' on signal (the 3rd bar, index 2)
        n_highlight = signal_bars[2].copy().set_stroke(GREEN_HIGHLIGHT, width=4)
        n_label = MathTex("n", color=GREEN_HIGHLIGHT)
        # Place n_label at D4 to avoid kernel_label (Issue 23 fix)
        self.place_at_grid(n_label, "D4")
        
        # Highlight index 'k' on kernel (the 1st bar, index 0)
        k_highlight = kernel_bars[0].copy().set_stroke(GREEN_HIGHLIGHT, width=4)
        k_label = MathTex("k", color=GREEN_HIGHLIGHT)
        self.place_at_grid(k_label, "F2")
        
        self.play(
            Create(n_highlight),
            Write(n_label),
            Create(k_highlight),
            Write(k_label)
        )
        self.wait(2)
