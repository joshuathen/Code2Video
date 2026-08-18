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
        self.setup_layout("Prerequisite: The Normal Distribution", [
            "The Normal Distribution forms a perfect bell curve.",
            "It is defined by its mean and standard deviation.",
            "The shape remains symmetrical around the central peak."
        ])
        
        # === Setup Elements ===
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 0.5, 0.1], axis_config={"include_numbers": False})
        bell_curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=WHITE)
        
        # Area within 1 std dev (x from -1 to 1)
        area = axes.get_area(bell_curve, x_range=[-1, 1], color="#33FF57", opacity=0.5)
        
        # Mean line
        mean_line = Line(axes.c2p(0, 0), axes.c2p(0, 0.4), color="#FF5733", stroke_width=4)
        
        # Labels
        x_label = Text("Value", font_size=20, color=WHITE)
        y_label = Text("Density", font_size=20, color=WHITE)
        
        # Place in grid
        plot_group = VGroup(axes, bell_curve, area, mean_line)
        self.place_in_area(plot_group, 'A2', 'D5', scale_factor=0.65)
        self.place_at_grid(y_label, 'B1', scale_factor=0.8)
        self.place_at_grid(x_label, 'F5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), Create(bell_curve))
        self.lecture[0].set_color("#FFD700")
        
        # === Animation for Lecture Line 2 ===
        self.play(Create(mean_line), FadeIn(area))
        self.lecture[1].set_color("#FF5733")
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(x_label), FadeIn(y_label))
        self.lecture[2].set_color("#33FF57")
        self.wait(2)
