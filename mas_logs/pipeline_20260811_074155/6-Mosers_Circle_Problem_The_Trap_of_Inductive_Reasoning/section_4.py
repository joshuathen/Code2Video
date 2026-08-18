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
        # Data from storyboard
        title = "The Mathematical Heartbreak (n=6)"
        lecture_lines = [
            "Now, let's try the sixth point very carefully.",
            "We connect all points, avoiding triple intersections.",
            "Let's count the resulting regions one by one.",
            "Instead of thirty-two, we only find thirty-one.",
            "The doubling pattern has finally broken."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Color definitions
        YELLOW = "#FFFF00"
        CYAN = "#00FFFF"
        MAGENTA = "#FF00FF"
        RED = "#FF0000"
        
        # Visual setup
        circle = Circle(radius=2.2, color=WHITE)
        # Fix for Issue 34: self.place_in_area(circle, 'B2', 'E5', scale_factor=0.9)
        self.place_in_area(circle, 'B2', 'E5', scale_factor=0.9)
        center = circle.get_center()
        # Actual radius after scaling
        eff_radius = 2.2 * 0.9
        
        # Angles in degrees, slightly irregular to avoid triple intersections
        angles = [15, 82, 137, 205, 255, 320]
        points = [center + np.array([eff_radius * np.cos(a * DEGREES), eff_radius * np.sin(a * DEGREES), 0]) for a in angles]
        dots = VGroup(*[Dot(p, color=YELLOW, radius=0.08) for p in points])
        
        # Chords calculation
        chords = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                chords.add(Line(points[i], points[j], color=CYAN, stroke_width=2))

        # === Animation for Lecture Line 1 ===
        # "Now, let's try the sixth point very carefully."
        self.lecture[0].set_color(YELLOW)
        self.play(Create(circle))
        self.play(FadeIn(dots[:5]))
        self.wait(0.5)
        # Highlight 6th point
        self.play(FadeIn(dots[5]), dots[5].animate.scale(1.5).set_color(YELLOW).scale(1/1.5))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We connect all points, avoiding triple intersections."
        self.lecture[1].set_color(CYAN)
        # Draw chords
        self.play(Create(chords, lag_ratio=0.05), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Let's count the resulting regions one by one."
        self.lecture[2].set_color(MAGENTA)
        
        # Generate 31 points inside the circle to represent regions
        region_markers = VGroup()
        for k in range(1, 32):
            # Using Golden Ratio spiral for distribution
            r_val = (eff_radius * 0.8) * np.sqrt((k - 0.5) / 31.0)
            theta_val = k * 137.5 * DEGREES
            pos = center + np.array([r_val * np.cos(theta_val), r_val * np.sin(theta_val), 0])
            
            m_dot = Dot(pos, color=MAGENTA, radius=0.04)
            m_label = Text(str(k), font_size=10, color=WHITE).move_to(pos + UP*0.1)
            region_markers.add(VGroup(m_dot, m_label))

        # Count 1-10
        self.play(LaggedStart(*[FadeIn(m) for m in region_markers[:10]], lag_ratio=0.2), run_time=2)
        # Count 11-25
        self.play(LaggedStart(*[FadeIn(m) for m in region_markers[10:25]], lag_ratio=0.15), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Instead of thirty-two, we only find thirty-one."
        self.lecture[3].set_color(RED)
        self.play(LaggedStart(*[FadeIn(m) for m in region_markers[25:]], lag_ratio=0.2), run_time=1.5)
        
        total_text = Text("Total: 31", font_size=36, weight=BOLD, color=RED)
        # Fix for Issue 32: self.place_in_area(total_text, 'F2', 'F5', scale_factor=1.0)
        self.place_in_area(total_text, 'F2', 'F5', scale_factor=1.0)
        self.play(Write(total_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The doubling pattern has finally broken."
        self.lecture[4].set_color(RED)
        failure_text = Text("31 ≠ 32", font_size=42, weight=BOLD, color=RED)
        # Fix for Issue 33: self.place_in_area(failure_text, 'A2', 'A5', scale_factor=1.1)
        self.place_in_area(failure_text, 'A2', 'A5', scale_factor=1.1)
        
        self.play(Write(failure_text))
        self.play(failure_text.animate.scale(1.3), rate_func=there_and_back, run_time=1.5)
        self.wait(2)
