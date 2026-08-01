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
        # Title and Lecture Lines
        title_text = "The Engine of Growth (Understanding 'e')"
        lines = [
            "e is the base of continuous, natural growth.",
            "Growth happens faster as the value increases.",
            "It represents change based on its current state."
        ]
        self.setup_layout(title_text, lines)

        # === Animation for Lecture Line 1 ===
        # Description: Display the letter 'e' (#55FF55) and the value '2.718...' directly below it.
        self.play(self.lecture[0].animate.set_color("#55FF55"))
        
        e_symbol = Text("e", slant=ITALIC, color="#55FF55")
        e_value = Text("2.718...", color="#55FF55")
        
        # Applied Issue 44: Place e_symbol at B4 with scale 1.5
        self.place_at_grid(e_symbol, "B4", scale_factor=1.5)
        # Applied Issue 43: Place e_value at C4 with scale 1.2
        self.place_at_grid(e_value, "C4", scale_factor=1.2)
        
        self.play(
            FadeIn(e_symbol, shift=UP * 0.3),
            Write(e_value),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Animate a line graph starting at (0,1) that curves upward sharply to demonstrate exponential growth.
        self.play(self.lecture[1].animate.set_color("#55AAFF"))
        
        axes = Axes(
            x_range=[0, 2.5, 1],
            y_range=[0, 10, 2],
            x_length=5,
            y_length=3.5,
            axis_config={"color": BLUE_B, "include_tip": True},
            tips=True
        )
        # Applied Issue 45: Place axes in area D1-F6
        self.place_in_area(axes, "D1", "F6", scale_factor=0.8)
        
        # Exponential curve: y = e^x
        graph = axes.plot(
            lambda x: np.exp(x),
            x_range=[0, 2.2],
            color="#55AAFF"
        )
        
        # Using MarkupText for the label to avoid MathTex/LaTeX requirement
        graph_label = axes.get_graph_label(graph, label=MarkupText("<i>e</i><sup>x</sup>"), x_val=2.2, direction=UR)
        
        self.play(Create(axes), run_time=1)
        self.play(Create(graph), Write(graph_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Flash the word 'GROWTH' (#55FF55) alongside the engine icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/engine.svg] 
        # in bold text next to the steepest part of the curve.
        self.play(self.lecture[2].animate.set_color("#55FF55"))
        
        # Applied Issue 37: Integration of engine icon asset
        engine_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/engine.svg")
        engine_icon.set_color("#55FF55")
        
        growth_text = Text("GROWTH", weight=BOLD, color="#55FF55")
        
        # Group icon and text for flash
        growth_group = VGroup(engine_icon, growth_text).arrange(RIGHT, buff=0.2).scale(0.6)
        
        # Position near the end of the curve (steepest part)
        steepest_point = axes.c2p(2.2, np.exp(2.2))
        growth_group.next_to(steepest_point, LEFT, buff=0.2)
        
        self.play(
            Flash(steepest_point, color="#55FF55", line_length=0.3, flash_radius=0.5),
            FadeIn(growth_group, scale=1.2),
            run_time=1.5
        )
        self.play(Indicate(growth_group, color="#55FF55"))
        self.wait(2)
