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
        # Setup layout
        lecture_lines = [
            "MLPs act as the internal library of the Transformer.",
            "They recognize patterns and inject relevant factual data.",
            "This turns a simple processor into a knowledgeable assistant."
        ]
        self.setup_layout("Summary & Conclusion", lecture_lines)

        # Colors for lines and corresponding graphics
        color_1 = BLUE_B
        color_2 = GREEN_B
        color_3 = GOLD_A

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))

        # W1 (Key Detector) and W2 (Value Injector)
        w1_box = Rectangle(width=2.5, height=1.5, color=color_1, fill_opacity=0.2)
        w1_text = Text("W1: Key Detector", font_size=18, color=color_1)
        w1_group = VGroup(w1_box, w1_text)
        
        w2_box = Rectangle(width=2.5, height=1.5, color=color_1, fill_opacity=0.2)
        w2_text = Text("W2: Value Injector", font_size=18, color=color_1)
        w2_group = VGroup(w2_box, w2_text)

        self.place_at_grid(w1_group, "B3", scale_factor=0.8)
        self.place_at_grid(w2_group, "D3", scale_factor=0.8)

        library_label = Text("Internal Library", font_size=24, color=color_1)
        self.place_at_grid(library_label, "A3", scale_factor=0.8)

        self.play(Create(w1_group), Create(w2_group), Write(library_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_2))

        # Lexi Robot Representation using Asset
        lexi = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg").set_color(color_2)
        self.place_at_grid(lexi, "C5", scale_factor=0.7)

        # Lexi identifying Paris
        prompt = Text("'Capital of France?'", font_size=16, color=WHITE)
        self.place_at_grid(prompt, "B5", scale_factor=0.7)
        
        answer = Text("'Paris!'", font_size=20, color=color_2)
        self.place_at_grid(answer, "D5", scale_factor=0.8)

        # Data flow animation
        arrow1 = Arrow(start=self.grid["B5"], end=self.grid["B3"], color=color_2, buff=0.4)
        arrow2 = Arrow(start=self.grid["B3"], end=self.grid["D3"], color=color_2, buff=0.4)
        arrow3 = Arrow(start=self.grid["D3"], end=self.grid["D5"], color=color_2, buff=0.4)

        self.play(FadeIn(lexi), Write(prompt))
        self.play(GrowArrow(arrow1))
        self.play(Indicate(w1_group))
        self.play(GrowArrow(arrow2))
        self.play(Indicate(w2_group))
        self.play(GrowArrow(arrow3))
        self.play(Write(answer))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_3))

        # Clear previous graphics for final credit
        self.play(
            FadeOut(w1_group), FadeOut(w2_group), FadeOut(library_label),
            FadeOut(lexi), FadeOut(prompt), FadeOut(answer),
            FadeOut(arrow1), FadeOut(arrow2), FadeOut(arrow3)
        )

        ending_text = Text("MLPs: The Knowledge Engines of AI", font_size=32, color=color_3)
        self.place_in_area(ending_text, "C2", "F5", scale_factor=0.9)

        self.play(Write(ending_text))
        self.play(Indicate(ending_text))
        self.wait(3)
