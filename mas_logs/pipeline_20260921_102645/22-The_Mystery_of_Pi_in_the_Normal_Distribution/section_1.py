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
        self.setup_layout("The Mystery of Pi in the Normal Distribution", [
            "The Gaussian function defines the bell curve.",
            "The normalization constant ensures total area is one.",
            "Why does π appear in the constant?"
        ])
        
        # Axes for the bell curve
        axes = Axes(x_range=[-3, 3], y_range=[0, 1], axis_config={"include_tip": False})
        bell_curve = axes.plot(lambda x: np.exp(-x**2/2), color="#FFD700")
        
        # === Animation for Lecture Line 1 ===
        # Fix: bell_curve positioned at A1-C6 to avoid overlap
        self.place_in_area(bell_curve, "A1", "C6", scale_factor=0.5)
        
        # Load asset: Bell icon
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        bell_icon.move_to(bell_curve.get_center())
        
        self.play(Create(bell_curve), FadeIn(bell_icon))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        const_label = MathTex(r"C = \frac{1}{\sqrt{2\pi}}", color="#FFFFFF")
        self.place_at_grid(const_label, "E2", scale_factor=0.8)
        self.play(Write(const_label))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        area = axes.get_area(bell_curve, x_range=[-3, 3], color="#00BFFF", opacity=0.3)
        self.place_in_area(area, "A1", "C6", scale_factor=0.5)
        self.play(FadeIn(area))
        
        # Fixes for crowding and clipping
        integral = MathTex(r"\int_{-\infty}^{\infty} f(x) dx = 1", color="#FFFFFF")
        self.place_in_area(integral, "D2", "E4", scale_factor=0.6)
        self.play(Write(integral))
        
        sigma = MathTex(r"\sigma", color="#FF4500")
        self.place_at_grid(sigma, "D5", scale_factor=0.6)
        self.play(Write(sigma))
        
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
