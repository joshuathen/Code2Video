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
        self.setup_layout("Mathematical Visualization (Malus's Law)", [
            "Malus's Law dictates the light's final intensity.",
            "Intensity follows an I zero cos squared theta pattern.",
            "Changing concentration shifts the intensity curve."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFC107")
        formula = MathTex(r"I = I_0 \\cos^2(\\theta)", color="#FFC107")
        # Fix for Issue 29/44: Move formula to A3
        self.place_at_grid(formula, "A3", scale_factor=1.0)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00BCD4")
        
        # Axes and Graph
        axes = Axes(x_range=[0, PI, PI/4], y_range=[0, 1.1, 0.5], axis_config={"include_numbers": False}).scale(0.5)
        graph = axes.plot(lambda x: (np.cos(x))**2, color="#00BCD4")
        graph_group = VGroup(axes, graph)
        # Fix for Issue 30/45: Adjust placement for graph
        self.place_in_area(graph_group, "C2", "E5", scale_factor=0.7)
        
        # Slider indicator (Asset integration Issue 18)
        # Position in Row B, Column 5
        slider = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg", color="#00BCD4")
        self.place_at_grid(slider, "B5", scale_factor=0.5)
        slider_label = Text("Angle", font_size=18, color="#00BCD4")
        slider_label.next_to(slider, DOWN, buff=0.1)
        
        # Dot updater
        theta_val = ValueTracker(0)
        dot = Dot(color="#00BCD4")
        dot.add_updater(lambda d: d.move_to(axes.c2p(theta_val.get_value(), (np.cos(theta_val.get_value()))**2)))
        
        self.play(Create(axes), Create(graph), FadeIn(dot), FadeIn(slider), Write(slider_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#4CAF50")
        self.play(theta_val.animate.set_value(PI), run_time=3, rate_func=linear)
        self.wait(1)
