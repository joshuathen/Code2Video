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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "AI creativity is high-dimensional statistical probability.",
            "Models navigate learned latent spaces.",
            "They don't copy, they generate anew."
        ]
        self.setup_layout("Conclusion and Real-world Impact", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # AI creativity is high-dimensional statistical probability.
        self.lecture[0].set_color("#00FA9A")
        # Load assets
        icons = VGroup(
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color="#00FA9A"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#00FA9A"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paintbrush.svg", color="#00FA9A"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#00FA9A")
        ).arrange(RIGHT, buff=0.2)
        self.place_in_area(icons, 'A4', 'C6', scale_factor=0.6)
        self.play(FadeIn(icons))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Models navigate learned latent spaces.
        self.lecture[1].set_color("#FF00FF")
        latent_progression = VGroup(*[
            Circle(radius=0.1 + i*0.05, color="#FF00FF") for i in range(5)
        ]).arrange(RIGHT)
        self.place_at_grid(latent_progression, 'D4', scale_factor=0.7)
        self.play(FadeIn(latent_progression))
        self.play(latent_progression.animate.shift(RIGHT * 2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # They don't copy, they generate anew.
        self.lecture[2].set_color("#FFFFFF")
        final_text = Text("The Future of Creativity", font_size=36, color="#FFFFFF")
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#FFFFFF")
        final_group = VGroup(final_text, computer_icon).arrange(DOWN)
        self.place_at_grid(final_group, 'E5', scale_factor=0.9)
        self.play(Write(final_text), FadeIn(computer_icon))
        self.wait(2)
