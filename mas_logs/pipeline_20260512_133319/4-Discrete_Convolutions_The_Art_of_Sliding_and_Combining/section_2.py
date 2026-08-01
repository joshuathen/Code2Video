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
        # Section title and lecture content
        title_text = "Prerequisite: Signals as Sequences"
        lecture_lines = [
            'Discrete signals are ordered sequences of numbers.',
            'Input x[n] meets an impulse response h[n].',
            'Visualize these sequences as bars on a graph.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color constants
        COLOR_X = "#0000FF" # Blue
        COLOR_H = "#FFA500" # Orange
        COLOR_TEXT = "#FFFFFF" # White

        # Helper: Build a signal visualization (bars + axis + label)
        def create_signal_vgroup(data, color, label_str):
            bars = VGroup()
            for val in data:
                # Scale height for visualization clarity
                h = val * 0.8
                rect = Rectangle(
                    height=h, 
                    width=0.4, 
                    fill_color=color, 
                    fill_opacity=0.7, 
                    stroke_color=WHITE, 
                    stroke_width=1
                )
                # Bottom-align the bar
                rect.move_to(UP * (h / 2))
                bars.add(rect)
            
            bars.arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
            
            # Horizontal Axis
            axis = Line(
                bars.get_left() + LEFT * 0.3, 
                bars.get_right() + RIGHT * 0.3, 
                color=WHITE
            ).next_to(bars, DOWN, buff=0)
            
            # Top Label
            tag = Text(label_str, font_size=20, color=COLOR_TEXT)
            tag.next_to(bars, UP, buff=0.4)
            
            return VGroup(bars, axis, tag)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Input x[n] = [1, 2, 1]
        x_signal = create_signal_vgroup([1, 2, 1], COLOR_X, "Input x[n]")
        # Place in top half of the grid (Issue #27 fix)
        self.place_in_area(x_signal, "A1", "C6", scale_factor=0.8)
        
        self.play(FadeIn(x_signal))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Kernel h[n] = [1, 0.5]
        h_signal = create_signal_vgroup([1, 0.5], COLOR_H, "Kernel h[n]")
        # Place in bottom half of the grid (Issue #28 fix)
        self.place_in_area(h_signal, "D1", "F6", scale_factor=0.8)
        
        self.play(FadeIn(h_signal))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Integer labels n=0, 1, 2...
        x_labels = VGroup()
        for i in range(len(x_signal[0])):
            lbl = Text(str(i), font_size=14, color=COLOR_TEXT)
            lbl.next_to(x_signal[0][i], DOWN, buff=0.1)
            x_labels.add(lbl)
            
        h_labels = VGroup()
        for i in range(len(h_signal[0])):
            lbl = Text(str(i), font_size=14, color=COLOR_TEXT)
            lbl.next_to(h_signal[0][i], DOWN, buff=0.1)
            h_labels.add(lbl)
            
        self.play(Write(x_labels), Write(h_labels))
        
        # Visual highlight: pulse bars to emphasize "bars on a graph"
        self.play(
            x_signal[0].animate.set_stroke(WHITE, width=3),
            h_signal[0].animate.set_stroke(WHITE, width=3),
            rate_func=there_and_back
        )
        self.wait(2)
        
        # Final cleanup: reset lecture highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
