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
            "Wordle is about reducing information state.",
            "Entropy transforms guessing into solving.",
            "Master architects systematically narrow choices."
        ]
        self.setup_layout("Conclusion: Summary and Intuition", lecture_lines)
        
        # Colors for lecture lines
        colors = ["#FFD700", "#00CED1", "#FF4500"]

        # Load assets
        keyboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg")
        monitor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg")

        # 1. Entropy measures uncertainty
        uncertainty_text = Text("Uncertainty = High Entropy", font_size=24, color="#FFD700")
        uncertainty_circle = Circle(radius=0.5, color="#FFD700", fill_opacity=0.3)
        uncertainty_group = VGroup(uncertainty_text, uncertainty_circle, keyboard_icon)
        uncertainty_group.arrange(DOWN, buff=0.2)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(uncertainty_group, 'A2', 'B4', scale_factor=0.9)
        self.play(self.lecture[0].animate.set_color(colors[0]), Write(uncertainty_group))
        self.wait(1)

        # 2. Information Gain drives the split decision
        gain_text = Text("Information Gain", font_size=24, color="#00CED1")
        self.place_at_grid(gain_text, 'C4', scale_factor=0.9)
        arrow = Arrow(start=self.grid["B3"], end=self.grid["D4"], color="#00CED1")
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]), FadeIn(gain_text), GrowArrow(arrow))
        self.wait(1)

        # 3. Decision Tree structure
        tree = VGroup(*[Dot(color="#FF4500") for _ in range(7)])
        tree.arrange_in_grid(rows=3, cols=3, buff=0.3)
        tree_with_monitor = VGroup(tree, monitor_icon)
        tree_with_monitor.arrange(DOWN, buff=0.2)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(tree_with_monitor, 'F4', scale_factor=0.7)
        self.play(self.lecture[2].animate.set_color(colors[2]), FadeIn(tree_with_monitor))
        self.wait(2)
