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
        title = "The Power of Two Delusion"
        lecture_lines = [
            "Let's test the pattern with five points.",
            "Connecting every pair creates exactly sixteen regions.",
            "It seems the count doubles with every point."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        YELLOW_PT = "#FFFF00"
        CYAN_CHORD = "#00FFFF"
        MAGENTA_NUM = "#FF00FF"
        GREEN_TEXT = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW_PT)
        
        # Circle setup
        circle = Circle(radius=2.0, color=WHITE)
        # Issue 25 Fix: Move circle to 'B3'-'E6' for better horizontal balance
        self.place_in_area(circle, 'B3', 'E6', scale_factor=0.8)
        
        circle_center = circle.get_center()
        # radius_val accounts for scale_factor=0.8 on Circle(radius=2.0)
        radius_val = 2.0 * 0.8
        
        # 5 Points
        points = VGroup()
        point_positions = []
        for i in range(5):
            angle = i * (360 / 5) * DEGREES + 90 * DEGREES
            pos = circle_center + np.array([np.cos(angle) * radius_val, np.sin(angle) * radius_val, 0])
            dot = Dot(pos, color=YELLOW_PT, radius=0.08)
            points.add(dot)
            point_positions.append(pos)
        
        # Chords
        chords = VGroup()
        for i in range(5):
            for j in range(i + 1, 5):
                line = Line(point_positions[i], point_positions[j], color=CYAN_CHORD, stroke_width=2)
                chords.add(line)
        
        self.play(Create(circle))
        self.play(LaggedStart(*[FadeIn(p) for p in points], lag_ratio=0.2))
        self.play(Create(chords), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(MAGENTA_NUM)
        
        # Approximation of 16 region centers for n=5
        region_centers = []
        
        # Center (1)
        region_centers.append(circle_center)
        
        # Inner ring of triangles (5)
        for i in range(5):
            angle = (i * 72 + 36) * DEGREES + 90 * DEGREES
            region_centers.append(circle_center + 0.5 * np.array([np.cos(angle), np.sin(angle), 0]))
            
        # Mid ring of quadrilaterals (5)
        for i in range(5):
            angle = (i * 72) * DEGREES + 90 * DEGREES
            region_centers.append(circle_center + 1.1 * np.array([np.cos(angle), np.sin(angle), 0]))
            
        # Outer ring of segments (5)
        for i in range(5):
            angle = (i * 72 + 36) * DEGREES + 90 * DEGREES
            region_centers.append(circle_center + 1.5 * np.array([np.cos(angle), np.sin(angle), 0]))

        region_labels = VGroup()
        for idx, pos in enumerate(region_centers):
            num = Text(str(idx + 1), font_size=18, color=MAGENTA_NUM)
            num.move_to(pos)
            region_labels.add(num)

        self.play(LaggedStart(*[Write(label) for label in region_labels], lag_ratio=0.15), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN_TEXT)
        
        confirmation_text = Text("2^(n-1) holds!", font_size=24, color=GREEN_TEXT)
        # Issue 26 Fix: Center confirmation text under the circle
        self.place_in_area(confirmation_text, 'F4', 'F5', scale_factor=0.8)
        
        self.play(Write(confirmation_text))
        self.wait(2)

        # Final cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
