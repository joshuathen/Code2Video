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
        self.setup_layout("Application: Why this matters in Data Science", [
            "Euclidean distance loses meaning in high dimensions.",
            "User preferences form points in 1,000-dimensional space.",
            "Most users appear equidistantly spaced."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualize high-dimensional data points using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/user.svg] as data representatives.
        user_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/user.svg", color=WHITE)
        dots = VGroup(*[user_icon.copy().set_height(0.2) for _ in range(15)])
        for dot in dots:
            dot.move_to(np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1, 1), 0]))
        
        # Applying fix for Issue 31
        self.place_in_area(dots, 'D2', 'F5', scale_factor=0.6)
        self.play(FadeIn(dots))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Display a distance metric calculation: d(x,y).
        # Applying fix for Issue 30 and 32 (A5 grid)
        dist_label = MathTex("d(x, y) = \\sqrt{\\sum (x_i - y_i)^2}", color=WHITE)
        self.place_at_grid(dist_label, 'A5', scale_factor=0.7)
        self.play(Write(dist_label))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Show how nearest neighbors become equidistant.
        # Select two points and highlight them
        p1 = dots[0].copy().set_color("#FF4500")
        p2 = dots[1].copy().set_color("#FFD700")
        # Ensure we don't scale or move beyond the existing group
        self.add(p1, p2)
        self.play(Indicate(p1), Indicate(p2))
        self.lecture[2].set_color(RED)
        self.wait(2)
