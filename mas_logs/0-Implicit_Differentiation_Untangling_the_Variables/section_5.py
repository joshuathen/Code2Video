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
        # Setup layout with title and lecture lines
        title = "Application: The Rollercoaster of Curves"
        lines = [
            "Some curves, like this infinity loop, are impossible to isolate.",
            "Implicit differentiation is our superpower for these complex paths.",
            "Meet Turbo the Snail racing along this intricate track.",
            "We calculate his exact direction at any given coordinate.",
            "This math keeps Turbo on track throughout the loop."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line in Purple
        self.lecture[0].set_color("#BF40BF")
        
        # Display Equation: (x^2 + y^2)^2 = x^2 - y^2
        # Use Text instead of MathTex to avoid latex issues
        equation = Text("(x^2 + y^2)^2 = x^2 - y^2", color="#BF40BF")
        # Fix for Issue 24: Adjust scale factor to 0.75
        self.place_in_area(equation, "A2", "A5", scale_factor=0.75)
        
        # Lemniscate of Bernoulli: (x^2+y^2)^2 = x^2-y^2
        # Parametric equations: x = cos(t)/(1+sin^2(t)), y = sin(t)cos(t)/(1+sin^2(t))
        lemniscate = ParametricFunction(
            lambda t: np.array([
                np.cos(t) / (1 + (np.sin(t)**2)),
                (np.sin(t) * np.cos(t)) / (1 + (np.sin(t)**2)),
                0
            ]),
            t_range=[0, 2*PI],
            color="#BF40BF"
        )
        # Fix for Issue 23: Adjust area to B2-F6 and scale factor to 1.8
        self.place_in_area(lemniscate, "B2", "F6", scale_factor=1.8)
        
        self.play(Write(equation), Create(lemniscate))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.lecture[1].set_color("#BF40BF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line in White
        self.lecture[2].set_color("#FFFFFF")
        
        # Create Turbo the Snail (Dot) and its label
        t_tracker = ValueTracker(0)
        turbo = Dot(color="#FFFFFF")
        turbo.move_to(lemniscate.point_from_proportion(0))
        
        turbo_label = Text("Turbo", font_size=18, color="#FFFFFF")
        turbo_label.next_to(turbo, UP, buff=0.15)
        
        # Updaters for Turbo and its label to follow the curve
        def turbo_updater(m):
            m.move_to(lemniscate.point_from_proportion(t_tracker.get_value()))
        
        def label_updater(m):
            m.next_to(turbo, UP, buff=0.15)
            
        turbo.add_updater(turbo_updater)
        turbo_label.add_updater(label_updater)
        
        self.play(FadeIn(turbo), FadeIn(turbo_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight fourth lecture line in Green
        self.lecture[3].set_color("#00FF00")
        
        # Dynamic Tangent Vector representing Turbo's direction
        def get_tangent_arrow():
            val = t_tracker.get_value()
            p = lemniscate.point_from_proportion(val)
            
            # Numerical approximation of the tangent direction
            eps = 0.005
            if val <= 1 - eps:
                p_ahead = lemniscate.point_from_proportion(val + eps)
                direction = p_ahead - p
            else:
                p_behind = lemniscate.point_from_proportion(val - eps)
                direction = p - p_behind
                
            # Normalize and scale the vector for visual clarity
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * 0.8
            else:
                direction = RIGHT * 0.1
                
            return Arrow(
                p, p + direction, 
                buff=0, 
                color="#00FF00", 
                stroke_width=4, 
                max_tip_length_to_length_ratio=0.3
            )

        tangent_arrow = always_redraw(get_tangent_arrow)
        self.play(Create(tangent_arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight fifth lecture line in White
        self.lecture[4].set_color("#FFFFFF")
        
        # Animate Turbo racing along the track using the t_tracker
        self.play(
            t_tracker.animate.set_value(1), 
            run_time=10, 
            rate_func=linear
        )
        self.wait(2)
