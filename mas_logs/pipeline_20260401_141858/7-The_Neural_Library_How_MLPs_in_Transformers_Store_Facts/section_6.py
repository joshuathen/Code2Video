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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        self.setup_layout("The Superposition: Storing Billions of Facts", [
            'How can one layer store millions of distinct facts?', 
            'High-dimensional spaces provide an incredible amount of room.', 
            'Vectors can be nearly perpendicular to avoid interference.', 
            'This phenomenon is known as superposition in neural networks.', 
            'Thousands of keys coexist without overlapping or blurring.'
        ])
        
        # Hex Colors for elements
        colors = ["#FFD700", "#00FFFF", "#ADFF2F", "#FF69B4", "#FFA500"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # A large circle (#555555) appears on screen.
        main_circle = Circle(radius=1.5, color="#555555", stroke_width=2)
        # Updated positioning per Issue 57
        self.place_in_area(main_circle, "B1", "F5", scale_factor=1.0)
        center_pt = main_circle.get_center()
        
        self.play(Create(main_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        # Representing high-dimensional space conceptually with the vast area within the circle.
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Two arrows at 90 degrees demonstrate orthogonality.
        vec1 = Arrow(start=center_pt, end=center_pt + RIGHT * 1.4, buff=0, color=colors[2])
        vec2 = Arrow(start=center_pt, end=center_pt + UP * 1.4, buff=0, color=colors[2])
        
        # Visual right angle marker
        square_marker = Square(side_length=0.2, color=colors[2], stroke_width=1)
        square_marker.move_to(center_pt + np.array([0.1, 0.1, 0]))
        
        self.play(GrowArrow(vec1), GrowArrow(vec2), FadeIn(square_marker))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # Dozens of new vectors (lines) are added to the circle.
        extra_vectors = VGroup()
        for i in range(24):
            angle = (i * 15) * DEGREES
            # Add a slight randomness to simulate "nearly" orthogonal/independent vectors
            offset = np.random.uniform(-0.1, 0.1)
            vec_line = Line(
                center_pt, 
                center_pt + np.array([np.cos(angle+offset), np.sin(angle+offset), 0]) * 1.4,
                stroke_width=1,
                color=colors[3]
            )
            extra_vectors.add(vec_line)
            
        # Text overlay: 'High-dimensional space allows for nearly orthogonal vectors.'
        overlay_text = Text(
            "High-dimensional space allows for\nnearly orthogonal vectors.", 
            font_size=18, 
            color=WHITE,
            line_spacing=0.8
        )
        # Updated positioning per Issue 56
        self.place_in_area(overlay_text, "A1", "A6", scale_factor=0.8)
        
        self.play(
            FadeIn(extra_vectors, lag_ratio=0.1),
            FadeIn(overlay_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        # The circle glows with thousands of distinct 'fact points'.
        fact_points = VGroup()
        for _ in range(200):
            # Uniform distribution inside the circle
            r = 1.4 * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            pos = center_pt + np.array([r * np.cos(theta), r * np.sin(theta), 0])
            dot = Dot(point=pos, radius=0.02, color=colors[4])
            fact_points.add(dot)
            
        # Create a glow effect
        glow_layer = main_circle.copy().set_stroke(color=colors[4], width=10, opacity=0.4)
        
        self.play(
            FadeOut(vec1), FadeOut(vec2), FadeOut(square_marker), FadeOut(extra_vectors),
            main_circle.animate.set_color(colors[4]),
            FadeIn(glow_layer),
            Create(fact_points, lag_ratio=0.005, run_time=2)
        )
        
        # Final glow pulse
        self.play(
            glow_layer.animate.scale(1.2).set_stroke(opacity=0),
            rate_func=rate_functions.ease_out_sine,
            run_time=1
        )
        self.wait(2)
