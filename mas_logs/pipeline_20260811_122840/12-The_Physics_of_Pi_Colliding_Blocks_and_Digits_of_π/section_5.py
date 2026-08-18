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
        self.setup_layout("Conclusion: The Elegance of Mechanics", 
                          ["Circular geometry governs the collision space.", 
                           "Pi naturally emerges from linear impacts.", 
                           "Physics reveals profound mathematical beauty."])
        
        # --- Animation Content ---
        
        # === Animation for Lecture Line 1 ===
        # Show a summary visualization of all collisions. [Color: #FFFFFF]
        circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle, 'B2', 'E4', scale_factor=0.85)
        self.play(Create(circle), self.lecture[0].animate.set_color("#FFFFFF"), run_time=2)

        # === Animation for Lecture Line 2 ===
        # Flash the final pi result alongside mass ratio. [Color: #FFFF00]
        pi_text = MathTex(r"\\pi \\approx 3.14159...", color=YELLOW)
        self.place_at_grid(pi_text, 'C4', scale_factor=0.7)
        self.play(FadeIn(pi_text), self.lecture[1].animate.set_color("#FFFF00"), run_time=2)

        # === Animation for Lecture Line 3 ===
        # Fade out all elements gracefully. [Color: #666666]
        self.play(
            FadeOut(circle), 
            FadeOut(pi_text), 
            self.lecture[2].animate.set_color("#666666"),
            run_time=2
        )
        self.wait(1)
