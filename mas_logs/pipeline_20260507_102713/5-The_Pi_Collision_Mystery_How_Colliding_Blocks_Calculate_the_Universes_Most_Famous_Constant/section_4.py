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
        # Updated lines to match mandatory Stage-3 script
        lines = [
            "A dot on the circle represents the system's state.",
            "Every physical collision reflects the point across the circle.",
            "Successive bounces create a zigzag path of state transitions."
        ]
        
        self.setup_layout("The Physics of a Bounce: A Reflection in State Space", lines)
        
        # Colors based on prompt and issues
        CIRCLE_COLOR = "#00FF00"  # Green
        DOT_COLOR = "#FFFFFF"     # White
        PATH_COLOR = "#FFFF00"    # Yellow
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CIRCLE_COLOR)
        
        # Setup Circle and Initial Point
        circle = Circle(radius=2.0, color=CIRCLE_COLOR)
        # Fix for Issue 32: Row B to F to avoid title
        self.place_in_area(circle, 'B2', 'F5')
        
        center = circle.get_center()
        
        # Initial point on circle circumference (0 degrees)
        start_angle = 0 * DEGREES
        p0 = center + np.array([np.cos(start_angle), np.sin(start_angle), 0]) * 2.0
        dot = Dot(p0, color=DOT_COLOR)
        
        self.play(Create(circle))
        self.play(FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(DOT_COLOR)
        
        # Angle indicator and label as per high-level design and Issue 33
        angle_indicator = Arc(radius=0.5, start_angle=0, angle=30*DEGREES, color=DOT_COLOR).move_to(center)
        angle_label = Text("θ", color=DOT_COLOR)
        # Fix for Issue 33: D4 position and scale 1.2
        self.place_at_grid(angle_label, 'D4', scale_factor=1.2)
        
        self.play(Create(angle_indicator), Write(angle_label))
        
        # First Reflection (Chord)
        reflect_angle_1 = 150 * DEGREES
        p1 = center + np.array([np.cos(reflect_angle_1), np.sin(reflect_angle_1), 0]) * 2.0
        chord1 = Line(p0, p1, color=DOT_COLOR, stroke_width=2)
        
        self.play(
            dot.animate.move_to(p1),
            Create(chord1),
            run_time=1.5
        )
        self.wait(0.5)
        self.play(FadeOut(angle_indicator), FadeOut(angle_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PATH_COLOR)
        
        # Rapid Bounces: Repeat reflection 5 times (total zigzag)
        # Angles selected to create a zigzag pattern
        angles = [300, 90, 240, 30, 180] 
        current_p = p1
        
        for a in angles:
            next_p = center + np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]) * 2.0
            new_chord = Line(current_p, next_p, color=PATH_COLOR, stroke_width=2)
            self.play(
                dot.animate.move_to(next_p),
                Create(new_chord),
                run_time=0.4
            )
            current_p = next_p
            
        self.wait(2)
