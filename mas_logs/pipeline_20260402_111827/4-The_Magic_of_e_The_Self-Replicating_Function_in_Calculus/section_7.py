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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "e is the natural language of growth.",
            "It is the function that is its own derivative.",
            "This constant remains the cornerstone of modern science."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFE0"))
        
        # 1. Rabbit (SVG Asset) - Issue 38
        rabbit = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/rabbit.svg", color=WHITE)
        self.place_at_grid(rabbit, "A2", scale_factor=0.6)
        rabbit_label = Text("Growth", font_size=18).next_to(rabbit, DOWN, buff=0.1)
        
        # 2. Coffee Cup (Simplified shape)
        cup_base = Arc(radius=0.4, start_angle=PI, angle=PI, color=WHITE)
        cup_top = Line(cup_base.get_start(), cup_base.get_end(), color=WHITE)
        cup = VGroup(cup_base, cup_top)
        self.place_at_grid(cup, "E2", scale_factor=0.8)
        cup_label = Text("Cooling", font_size=18).next_to(cup, DOWN, buff=0.1)
        
        # 3. Graph curve (e^x) - Issue 55 (move to C2)
        axes = Axes(x_range=[0, 2, 1], y_range=[0, 4, 1], x_length=1.5, y_length=1.5, 
                   axis_config={"include_tip": False, "font_size": 12})
        curve = axes.plot(lambda x: np.exp(x-1), x_range=[0, 2], color=BLUE)
        graph = VGroup(axes, curve)
        self.place_at_grid(graph, "C2", scale_factor=0.8)
        graph_label = Text("Calculus", font_size=18).next_to(graph, DOWN, buff=0.1)

        self.play(
            FadeIn(rabbit), Write(rabbit_label),
            Create(cup), Write(cup_label),
            Create(graph), Write(graph_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E0FFE0"))
        
        # Group everything and move to center area
        everything = VGroup(rabbit, rabbit_label, cup, cup_label, graph, graph_label)
        target_center = self.grid["C3"] # Approximation for C3-D4 center
        
        self.play(
            everything.animate.scale(0.5).move_to(target_center),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Glowing gold 'e' - Issue 56 (move to B4-D5)
        e_symbol = Text("e", color="#FFD700").scale(4)
        self.place_in_area(e_symbol, "B4", "D5")
        
        # Natural Base Text - Issue 57 (move to E4-F6)
        natural_base_text = Text("The Natural Base", font_size=32, color="#FFD700")
        self.place_in_area(natural_base_text, "E4", "F6")

        self.play(
            FadeOut(everything),
            FadeIn(e_symbol, shift=UP),
            Write(natural_base_text)
        )
        
        # Glowing effect
        self.play(
            Indicate(e_symbol, color="#FFD700", scale_factor=1.2),
            Indicate(natural_base_text, color="#FFD700"),
            run_time=2
        )
        self.wait(3)
