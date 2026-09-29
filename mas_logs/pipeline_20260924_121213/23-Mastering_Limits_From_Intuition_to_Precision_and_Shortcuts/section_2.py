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
        lecture_lines = [
            "Define epsilon for target accuracy.",
            "Define delta for input range.",
            "Does delta force epsilon bounds?",
            "The box trap proves it.",
            "Precision ensures the function limit."
        ]
        self.setup_layout("The Formal Rigor: Epsilon-Delta Definition", lecture_lines)
        
        # Define axes and function
        axes = Axes(x_range=[-2, 4], y_range=[-1, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.65)
        self.add(axes)
        
        func = axes.plot(lambda x: 0.5 * (x - 1)**2 + 1, color="#00FFFF")
        self.add(func)
        
        # Epsilon-Delta elements
        epsilon_val = 0.5
        delta_val = 0.6
        c = 1.0
        L = 1.0
        
        # Using SVG for the Epsilon-box
        box_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg"
        epsilon_rect = SVGMobject(box_path, color="#FFFF00")
        epsilon_rect.set_width(1.5)
        epsilon_rect.move_to(axes.c2p(c, L))
        
        # Delta interval
        delta_line = Line(axes.c2p(c - delta_val, -0.1), axes.c2p(c + delta_val, -0.1), color="#00FF00", stroke_width=4)
        
        # Labels
        eps_label = Text("Epsilon", font_size=18, color="#FFFF00")
        self.place_at_grid(eps_label, 'C1', scale_factor=0.7)
        
        del_label = Text("Delta", font_size=18, color="#00FF00")
        self.place_at_grid(del_label, 'F3', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Create(epsilon_rect))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), Create(delta_line))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE), FadeIn(epsilon_rect))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Shrink demonstration
        self.play(
            epsilon_rect.animate.scale(0.5),
            delta_line.animate.scale(0.5)
        )
        self.wait(1)
