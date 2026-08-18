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
        lines = [
            "Imagine a rabbit jumping halfway toward a carrot.",
            "Each jump brings the rabbit closer to x=2.",
            "The distance halves, yet it never reaches two.",
            "We define the limit as its target position.",
            "A limit describes behavior near a point."
        ]
        self.setup_layout("The Intuitive Foundation: The 'Zeno's Rabbit' Approach", lines)

        # Setup Number Line
        nl = NumberLine(x_range=[0, 3, 1], length=5, include_numbers=True)
        self.place_at_grid(nl, 'C2')
        self.add(nl)

        # Carrot
        carrot = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/carrot.png")
        carrot.set_color("#FF8C00")
        self.place_at_grid(carrot, 'C5', scale_factor=0.3)
        self.add(carrot)

        # Rabbit
        rabbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg")
        rabbit.set_color("#8B4513")
        self.place_at_grid(rabbit, 'C1', scale_factor=0.3)
        rabbit.next_to(nl.number_to_point(0), UP)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(rabbit), self.lecture[0].animate.set_color("#FF8C00"))

        # === Animation for Lecture Line 2 ===
        self.play(rabbit.animate.move_to(nl.number_to_point(1) + UP*0.2), self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.play(
            rabbit.animate.move_to(nl.number_to_point(1.5) + UP*0.2),
            run_time=0.5
        )
        self.play(
            rabbit.animate.move_to(nl.number_to_point(1.75) + UP*0.2),
            run_time=0.5
        )
        self.play(self.lecture[2].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 4 ===
        limit_line = DashedLine(nl.number_to_point(2) + DOWN*0.2, nl.number_to_point(2) + UP*0.5, color=WHITE)
        limit_text = Text("Limit", font_size=18, color=WHITE).next_to(limit_line, UP)
        self.play(Create(limit_line), Write(limit_text), self.lecture[3].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 5 ===
        glow = Dot(rabbit.get_center(), color="#FFD700", radius=0.3, fill_opacity=0.3)
        self.play(FadeIn(glow), rabbit.animate.set_color("#FFD700"), self.lecture[4].animate.set_color("#00FF00"))
        self.wait(2)
