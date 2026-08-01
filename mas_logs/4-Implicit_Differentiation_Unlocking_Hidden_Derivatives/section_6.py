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
        # Initial layout setup
        lecture_lines = [
            "Every y derivative needs a dy/dx tail.",
            "Implicit differentiation unlocks geometry for complex curves.",
            "Now you can find slopes for any hidden function."
        ]
        self.setup_layout("Summary & Visual Review", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Every y derivative needs a dy/dx tail.
        # Use Gold color #FFD700
        GOLD_COLOR = "#FFD700"
        self.lecture[0].set_color(GOLD_COLOR)
        
        # Display the rule: y derivative -> add dy/dx
        rule_text = Text("y derivative -> add dy/dx", color=GOLD_COLOR)
        self.place_in_area(rule_text, "A1", "B6", scale_factor=1.1)
        
        self.play(Write(rule_text))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Implicit differentiation unlocks geometry for complex curves.
        # Use Pink color #FFC0CB
        PINK_COLOR = "#FFC0CB"
        self.lecture[1].set_color(PINK_COLOR)
        
        # Heart-shaped curve (Cardioid-like parametric)
        heart = ParametricFunction(
            lambda t: np.array([
                16 * np.sin(t)**3,
                13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t),
                0
            ]),
            t_range=[0, 2 * PI],
            color=PINK_COLOR
        )
        # Pre-scale heart because the original values are large
        heart.scale(0.06)
        self.place_in_area(heart, "D1", "E3")
        
        # Infinity symbol (Lemniscate of Bernoulli)
        infinity = ParametricFunction(
            lambda t: np.array([
                2.8 * np.cos(t) / (1 + np.sin(t)**2),
                2.8 * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2),
                0
            ]),
            t_range=[0, 2 * PI],
            color=PINK_COLOR
        )
        self.place_in_area(infinity, "D4", "E6")
        
        # Labels for the shapes
        heart_label = Text("Heart Curve", font_size=18, color=PINK_COLOR)
        self.place_at_grid(heart_label, "C2", scale_factor=0.8)
        
        inf_label = Text("Infinity Curve", font_size=18, color=PINK_COLOR)
        self.place_at_grid(inf_label, "C5", scale_factor=0.8)
        
        self.play(
            Create(heart), 
            Create(infinity),
            FadeIn(heart_label),
            FadeIn(inf_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Now you can find slopes for any hidden function.
        # Animate tangent lines rotating around the curves.
        self.lecture[2].set_color(WHITE)
        
        t_tracker = ValueTracker(0)
        
        # Create tangent lines using AlwaysRedraw for dynamic movement
        heart_tangent = always_redraw(lambda: TangentLine(
            heart, 
            t_tracker.get_value(), 
            length=1.4, 
            color=WHITE
        ))
        
        inf_tangent = always_redraw(lambda: TangentLine(
            infinity, 
            t_tracker.get_value(), 
            length=1.4, 
            color=WHITE
        ))
        
        self.add(heart_tangent, inf_tangent)
        
        # Fast rotation around the perimeter to show the "unlocking" of geometry
        self.play(
            t_tracker.animate.set_value(1),
            run_time=8,
            rate_func=linear
        )
        
        # Hold on final frame
        self.wait(3)
