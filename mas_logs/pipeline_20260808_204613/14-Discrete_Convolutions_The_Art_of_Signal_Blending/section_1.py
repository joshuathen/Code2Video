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
        lecture_lines = [
            "A clap echoes in a concert hall.",
            "The hall's geometry defines the echo.",
            "This echo is a mathematical convolution.",
            "Inputs blend with the system's memory.",
            "We see this through signal reverberation."
        ]
        self.setup_layout("Intuitive Hook: The Echo Effect", lecture_lines)
        
        # Define assets
        hall_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hall.svg")
        self.place_in_area(hall_icon, "B2", "C5", scale_factor=0.8)
        self.add(hall_icon)
        
        # Define signals (right-aligned to columns 4-6)
        axes = Axes(x_range=[0, 10, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        signal1 = axes.plot(lambda x: np.sin(x*2) * np.exp(-x/5), color="#00FFFF")
        signal2 = axes.plot(lambda x: np.sin((x-2)*2) * np.exp(-(x-2)/5) * 0.5, color="#FF00FF")
        
        signal_group = VGroup(axes, signal1, signal2)
        # Place in D2-E5 but shift right manually for columns 4-6 alignment within the grid logic
        self.place_in_area(signal_group, "D4", "E6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Create(axes), Create(signal1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.play(Create(signal2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(Indicate(signal_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.play(FadeOut(signal_group), FadeOut(hall_icon))
        self.wait(1)
