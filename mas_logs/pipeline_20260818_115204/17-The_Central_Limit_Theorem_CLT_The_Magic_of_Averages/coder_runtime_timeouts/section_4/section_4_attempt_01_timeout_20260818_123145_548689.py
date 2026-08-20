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
        self.setup_layout("The Magic of Averages", ["Watch the histogram morph over time.", "As samples increase, patterns stabilize.", "Chaos collapses into symmetry."])
        
        # Histogram setup
        bins = 20
        # Start with a skewed distribution
        data = np.random.beta(2, 5, 1000)
        hist = VGroup(*[Rectangle(width=0.3, height=0.1, fill_opacity=0.8, color="#D35400", stroke_width=0) for _ in range(bins)])
        hist.arrange(RIGHT, buff=0.05, aligned_edge=DOWN)
        self.place_in_area(hist, 'B1', 'E6', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(hist), self.lecture[0].animate.set_color("#D35400"))
        
        # === Animation for Lecture Line 2 ===
        # Morphing over samples (simulated by updating bar heights)
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#7F8C8D"))
        
        def update_hist(mob, alpha):
            # As alpha goes 0 -> 1, morph toward Normal distribution
            # Simple linear interpolation for bar heights
            target_data = np.random.normal(0.5, 0.1, 1000)
            target_hist, _ = np.histogram(target_data, bins=bins, range=(0,1))
            
            # Use current data as Beta(2, 5)
            current_hist, _ = np.histogram(data, bins=bins, range=(0,1))
            
            for i, bar in enumerate(mob):
                h = (1-alpha) * (current_hist[i]/50) + alpha * (target_hist[i]/50)
                bar.stretch_to_height(h)
                bar.set_color(interpolate_color(ManimColor.parse("#D35400"), ManimColor.parse("#27AE60"), alpha))
        
        self.play(UpdateFromAlphaFunc(hist, update_hist), run_time=3)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#27AE60"))
        self.wait(1)
