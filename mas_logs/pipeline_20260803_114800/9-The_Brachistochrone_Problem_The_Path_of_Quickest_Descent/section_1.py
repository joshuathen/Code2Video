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
        # Data from storyboard
        title_text = "The Counter-Intuitive Challenge"
        lecture_lines = [
            "What path is the fastest between two points?",
            "A straight line is the shortest distance, not time.",
            "Curved paths often allow objects to reach destinations sooner."
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_A = "#FFD700"
        COLOR_B = "#00BFFF"
        COLOR_STRAIGHT = "#FFFFFF"
        COLOR_CURVE = "#FF4500"
        COLOR_FLASH = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "What path is the fastest between two points?"
        self.play(self.lecture[0].animate.set_color(COLOR_A))
        
        # 1. Define Points A and B (Fixing Issue 27 and 28)
        dot_a = Dot(color=COLOR_A)
        dot_b = Dot(color=COLOR_B)
        label_a = Text("A", font_size=24, color=COLOR_A)
        label_b = Text("B", font_size=24, color=COLOR_B)
        
        self.place_at_grid(dot_a, "B4", scale_factor=0.8) # Issue 27
        self.place_at_grid(dot_b, "E6", scale_factor=0.8) # Issue 28
        label_a.next_to(dot_a, UP, buff=0.1)
        label_b.next_to(dot_b, DOWN, buff=0.1)

        self.play(
            FadeIn(dot_a), FadeIn(label_a),
            FadeIn(dot_b), FadeIn(label_b),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A straight line is the shortest distance, not time."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_STRAIGHT)
        )
        
        # 2. Define Paths
        start_pos = self.grid["B4"]
        end_pos = self.grid["E6"]
        
        straight_path = Line(start_pos, end_pos, color=COLOR_STRAIGHT)
        
        # Curve: Cubic Bezier to simulate a faster descent curve
        # Adjust control point to B5/E4 to avoid swinging left too much
        # We want it to go down then right. 
        # Control 1: Column 4, Row E (drops vertically from B4)
        # Control 2: Column 4, Row E 
        control_pos = self.grid["E4"] 
        curved_path = CubicBezier(start_pos, control_pos, control_pos, end_pos, color=COLOR_CURVE)

        self.play(Create(straight_path), run_time=1)
        
        # 3. Define Sliders Pip and Pop
        pip = Dot(color=COLOR_STRAIGHT).scale(1.2)
        pip_label = Text("Pip", font_size=18, color=COLOR_STRAIGHT)
        
        pop = Dot(color=COLOR_CURVE).scale(1.2)
        pop_label = Text("Pop", font_size=18, color=COLOR_CURVE)

        # Trackers for movement
        pip_tracker = ValueTracker(0)
        pop_tracker = ValueTracker(0)

        # Updaters for position
        pip.add_updater(lambda m: m.move_to(straight_path.point_from_proportion(pip_tracker.get_value())))
        pip_label.add_updater(lambda m: m.next_to(pip, UP, buff=0.1))
        
        pop.add_updater(lambda m: m.move_to(curved_path.point_from_proportion(pop_tracker.get_value())))
        pop_label.add_updater(lambda m: m.next_to(pop, DOWN, buff=0.1))

        self.add(pip, pip_label)
        self.play(pip_tracker.animate.set_value(0.3), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Curved paths often allow objects to reach destinations sooner."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_CURVE)
        )
        self.play(Create(curved_path), run_time=1)
        self.add(pop, pop_label)
        self.wait(0.5)

        # Race Sequence
        # Reset Pip
        self.play(pip_tracker.animate.set_value(0), run_time=0.5)
        self.wait(0.5)

        # Execution of the race: Pop finishes in 2s, Pip still halfway
        self.play(
            pip_tracker.animate(rate_func=linear).set_value(0.6),
            pop_tracker.animate(rate_func=rate_functions.ease_in_quad).set_value(1),
            run_time=2
        )
        
        # Pop reaches B first; curved path flashes yellow
        self.play(curved_path.animate.set_color(COLOR_FLASH), run_time=0.2)
        self.play(curved_path.animate.set_color(COLOR_CURVE), run_time=0.2)
        self.play(curved_path.animate.set_color(COLOR_FLASH), run_time=0.2)
        self.play(curved_path.animate.set_color(COLOR_CURVE), run_time=0.2)
        
        # Pip finishes
        self.play(pip_tracker.animate(rate_func=linear).set_value(1), run_time=1)
        self.wait(2)
