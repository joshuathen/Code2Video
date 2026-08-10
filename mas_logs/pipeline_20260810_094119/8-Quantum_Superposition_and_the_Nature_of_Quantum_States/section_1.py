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
            "Classical bits are either zero or one.",
            "Quantum systems allow simultaneous states.",
            "Imagine a dial, not a switch.",
            "This is a complex vector space.",
            "The qubit is our fundamental unit."
        ]
        self.setup_layout("The Classical vs. Quantum Divide", lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        switch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg", color="#FF4500")
        bit0 = switch.copy().scale(1.5)
        bit1 = switch.copy().scale(1.5)
        self.place_at_grid(bit0, 'B2', scale_factor=1.2)
        self.place_at_grid(bit1, 'B5', scale_factor=1.2)
        self.play(FadeIn(bit0), FadeIn(bit1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        plane = NumberPlane(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(plane, 'C2', 'D5', scale_factor=0.8)
        vec = Vector([1, 0.5], color="#00CED1")
        vec.shift(plane.get_center())
        self.play(Create(plane), GrowArrow(vec))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        dial = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dial.svg", color="#FFFF00")
        self.place_in_area(dial, 'C2', 'D5', scale_factor=0.5)
        vec.set_color("#FFFF00")
        self.play(vec.animate.put_start_and_end_on(plane.get_center(), plane.c2p(0.5, 0.866)), FadeIn(dial))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        circle = Circle(radius=plane.c2p(1,0)[0]-plane.c2p(0,0)[0], color="#FFFFFF").move_to(plane.get_center())
        self.play(Create(circle))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        label = Text("Qubit", color="#00FF00").scale(0.8)
        self.place_at_grid(label, 'E3', scale_factor=1.0)
        self.play(Write(label))
        self.wait(2)
