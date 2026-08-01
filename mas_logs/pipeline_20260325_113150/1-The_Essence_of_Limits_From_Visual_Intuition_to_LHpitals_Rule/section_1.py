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
        # Setup layout with title and lecture lines
        title_text = "The Intuition: The 'Almost' Point"
        lecture_lines = [
            "A limit describes where a function is headed.",
            "Imagine an ant walking along this graph.",
            "At x equals 1, there is a tiny pothole.",
            "The ant can't step there, but follows the path.",
            "Its destination is clearly at height y equals 2."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Replaced MathTex with Text to avoid LaTeX dependency error
        # Fix Issue 24: Move formula to A3-A6 to align with axes
        formula = Text("f(x) = (x^2 - 1) / (x - 1)", color="#FFFFFF")
        self.place_in_area(formula, "A3", "A6", scale_factor=0.8)
        
        self.play(
            Write(formula),
            self.lecture[0].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix Issue 23: Move axes to C3-F6 to avoid crowding with text
        axes = Axes(
            x_range=[0, 2.5, 1],
            y_range=[0, 3.5, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": "#FFFFFF"}
        )
        self.place_in_area(axes, "C3", "F6", scale_factor=0.8)
        
        # Simplified function line
        line_graph = axes.plot(lambda x: x + 1, x_range=[0, 2.2], color="#FFFFFF")
        
        self.play(
            Create(axes),
            Create(line_graph),
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reveal a small #FF0000 hollow circle (the hole) at the point (1, 2)
        hole = Circle(radius=0.1, color="#FF0000", fill_opacity=0).move_to(axes.c2p(1, 2))
        
        self.play(
            Create(hole),
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Fix Issue 21: Use Robo-Ant [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/ant.svg]
        # Robo-Ant moving smoothly along the line towards x = 1
        ant = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ant.svg")
        ant.set_color("#00FF00")
        ant.scale(0.2)
        ant.move_to(axes.c2p(0.5, 1.5))
        
        self.play(
            FadeIn(ant),
            self.lecture[2].animate.set_color("#FFFFFF"),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        self.play(
            ant.animate.move_to(axes.c2p(0.92, 1.92)),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Indicator line and label for y = 2
        indicator_line = DashedLine(
            start=axes.c2p(1, 2),
            end=axes.c2p(0, 2),
            color="#FFFF00"
        )
        # Replaced MathTex with Text to avoid LaTeX dependency error
        y_label = Text("y = 2", color="#FFFF00", font_size=20).next_to(axes.c2p(0, 2), LEFT, buff=0.1)
        
        self.play(
            Create(indicator_line),
            Write(y_label),
            self.lecture[3].animate.set_color("#FFFFFF"),
            self.lecture[4].animate.set_color("#FFFF00")
        )
        self.wait(2)
