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
        # Define Colors
        POINT_COLOR = "#E0FFFF" # Light Cyan
        FATOU_COLOR = "#00008B" # Dark Blue (as requested by hex)
        
        lecture_lines = [
            "Some regions exhibit predictable and stable behavior.",
            "This stable territory is called the Fatou Set.",
            "Nearby points move together like a calm pond."
        ]
        
        self.setup_layout("Stability vs. Chaos: The Fatou Set", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show two light cyan (#E0FFFF) points placed very close together.
        self.lecture[0].set_color(POINT_COLOR)
        
        dot1 = Dot(color=POINT_COLOR, radius=0.08)
        dot2 = Dot(color=POINT_COLOR, radius=0.08)
        dots = VGroup(dot1, dot2).arrange(RIGHT, buff=0.15)
        
        # Place at grid C4 (optimized position to avoid lecture text)
        self.place_at_grid(dots, "C4", scale_factor=1.0)
        
        self.play(FadeIn(dots))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Color the background region behind them light blue (#00008B) labeled 'Fatou Set'.
        self.lecture[1].set_color(FATOU_COLOR)
        
        # Region covering central part of the grid
        fatou_region = RoundedRectangle(
            corner_radius=0.2, 
            width=3.5, 
            height=3.5, 
            fill_color=FATOU_COLOR, 
            fill_opacity=0.4,
            stroke_color=FATOU_COLOR,
            stroke_width=2
        )
        # Positioned to avoid crowding lecture notes
        self.place_in_area(fatou_region, "B3", "E6", scale_factor=1.0)
        
        fatou_label = Text("Fatou Set", font_size=24, color=WHITE)
        # Use area-based positioning for better centering
        self.place_in_area(fatou_label, "B4", "B5", scale_factor=0.8)
        
        self.play(
            FadeIn(fatou_region),
            Write(fatou_label)
        )
        # Ensure dots remain on top of the newly added background
        dots.set_z_index(10)
        self.add(dots)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Move both points along identical smooth, circular paths simultaneously.
        self.lecture[2].set_color(POINT_COLOR)
        
        # Circular path rotation around a pivot near the group to show stability
        # The points move together, maintaining their relative distance.
        pivot = dots.get_center() + DOWN * 0.3
        
        self.play(
            Rotate(dots, angle=2*PI, about_point=pivot, rate_func=linear, run_time=4)
        )
        self.wait(2)
