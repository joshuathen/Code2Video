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
            "Discrete signals are sequences of values like pixel intensities.",
            "We use a kernel as a sliding window tool.",
            "Kernels transform target pixels based on their neighbors.",
            "Imagine a grid representing part of a cat.",
            "We calculate average brightness to smooth the image."
        ]
        self.setup_layout("Prerequisites & Intuition", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display 3x3 grid of values
        pixel_grid = VGroup(*[Text(str(i), font_size=20) for i in range(1, 10)]).arrange_in_grid(3, 3, buff=0.3)
        self.place_in_area(pixel_grid, 'A2', 'C4')
        self.play(FadeIn(pixel_grid), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Highlight a 3x3 kernel sliding window frame
        kernel = Square(color="#FFD700", side_length=2.5)
        self.place_in_area(kernel, 'A2', 'C4')
        self.play(Create(kernel), self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        # Kernels transform target pixels
        self.play(Indicate(pixel_grid), self.lecture[2].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 4 ===
        # Visualize cat asset on 5x5 grid
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        self.place_in_area(cat_icon, 'D1', 'F6', scale_factor=0.3)
        self.play(FadeIn(cat_icon), self.lecture[3].animate.set_color("#FF69B4"))

        # === Animation for Lecture Line 5 ===
        # Display smoothing calculation output
        avg_text = Text("Avg: 5.5", font_size=24, color="#32CD32")
        self.place_at_grid(avg_text, 'D5')
        self.play(Write(avg_text), self.lecture[4].animate.set_color("#32CD32"))
        self.wait(2)
