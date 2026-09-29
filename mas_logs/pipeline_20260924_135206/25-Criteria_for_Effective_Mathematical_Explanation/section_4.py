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
        lines = [
            "Relatability anchors concepts in long-term memory.",
            "Connect abstract math to real-world scenarios.",
            "Squirrel climbing trees models velocity rates."
        ]
        self.setup_layout("Criterion 3: Relatability and Application", lines)
        
        # Define visual objects
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        formula = MathTex(r"v = \frac{ds}{dt}", color="#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(gear, "B2", scale_factor=1.0) # ID 30: Scale 1.0
        self.play(Create(gear))
        self.wait(0.5)
        self.place_at_grid(formula, "B4", scale_factor=1.2) # ID 29: Grid B4, Scale 1.2
        self.play(Transform(gear, formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        tree = Line(start=self.grid["F4"], end=self.grid["C4"], color=GRAY)
        squirrel = Dot(color=ORANGE)
        
        animation_group = VGroup(tree, squirrel)
        self.place_in_area(animation_group, "C3", "E5", scale_factor=0.9) # ID 31: Area C3-E5, Scale 0.9
        
        self.add(tree)
        self.play(FadeIn(squirrel))
        self.play(squirrel.animate.move_to(tree.get_end()), run_time=2)
        self.wait(1)
