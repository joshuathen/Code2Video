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
        self.setup_layout("The Power of Sample Size (n)", [
            "Sample size, n, is our magic number.",
            "Small samples reveal the original messy distribution.",
            "Larger samples narrow the spread of means."
        ])

        # Prepare base objects
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False})
        
        # Assets (Using generic placeholders as per instructions if no file found)
        # Note: The provided paths are /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        n_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Curves
        curve1 = axes.plot(lambda x: np.exp(-(x**2)/2) / np.sqrt(2*np.pi), color="#FF5733") # n=2
        curve2 = axes.plot(lambda x: np.exp(-(x**2)/(2*0.1)) / np.sqrt(2*np.pi*0.1), color="#33FF57") # n=30

        group = VGroup(axes, curve1)
        self.place_in_area(group, 'B3', 'E6', scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        n_label = Text("n", font_size=40, color="#FFFFFF")
        self.place_at_grid(n_label, "A3")
        self.play(FadeIn(n_label), FadeIn(n_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        self.play(Create(group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        curve2.match_style(curve1)
        self.play(Transform(curve1, curve2))
        
        label = Text("n=30", font_size=20, color="#33FF57")
        self.place_at_grid(label, 'B5', scale_factor=0.75)
        self.play(Write(label))
        self.wait(2)
