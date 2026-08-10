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
        lecture_lines = [
            "The Born rule governs probability calculations.",
            "The squares of amplitudes must sum to one.",
            "Measurement forces the system into one basis state.",
            "Schrödinger's cat provides our intuitive analogy.",
            "Opening the box collapses the wavefunction instantly."
        ]
        self.setup_layout("Measurement: The Collapse of the Wavefunction", lecture_lines)
        
        # Load Assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        box_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")

        # Elements
        born_rule = MathTex(r"|\alpha|^2 + |\beta|^2 = 1", color=ORANGE)
        # Fix: Use Group because ImageMobject is not a VMobject
        born_group = Group(born_rule, cat_icon).arrange(RIGHT)

        superposition_visual = VGroup(
            Circle(radius=0.5, color=BLUE, fill_opacity=0.5),
            box_icon
        )
        
        sharp_state = Dot(radius=0.2, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE)
        self.place_at_grid(born_group, 'B4', scale_factor=0.6)
        self.play(FadeIn(born_group))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        self.place_in_area(superposition_visual, 'C2', 'D5', scale_factor=0.9)
        self.play(FadeIn(superposition_visual))
        self.play(Rotate(superposition_visual, angle=2*PI, run_time=2))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        flash = Rectangle(width=8, height=6, color=WHITE, fill_opacity=0.5)
        self.place_at_grid(camera_icon, 'A6', scale_factor=0.5)
        self.play(FadeIn(camera_icon))
        
        self.place_at_grid(sharp_state, 'E3', scale_factor=1.0)
        self.play(
            FadeOut(superposition_visual),
            FadeIn(sharp_state)
        )
        self.play(FadeIn(flash), FadeOut(flash), run_time=0.5)
        self.wait(2)
