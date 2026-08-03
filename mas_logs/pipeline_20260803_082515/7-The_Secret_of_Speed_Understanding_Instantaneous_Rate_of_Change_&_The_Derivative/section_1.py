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
        self.setup_layout("The Mystery of the Speed Trap", [
            "Dash the cheetah sprints 100 meters in 4 seconds.",
            "His average speed is exactly 25 meters per second.",
            "We freeze-frame his journey at exactly 2.5 seconds.",
            "Look closely at the position of his paws now.",
            "How fast is he moving at this specific instant?"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Dash the cheetah sprints 100 meters in 4 seconds.
        # Color: Yellow (#FFFF00)
        self.lecture[0].set_color("#FFFF00")
        
        dash = Circle(radius=0.3, color="#FFFF00", fill_opacity=1)
        # Issue 34 fix: Use C4 as the placement point as requested
        self.place_at_grid(dash, "C4")
        
        path_start = self.grid["C1"]
        path_end = self.grid["C6"]
        
        # Use ValueTracker for horizontal movement
        pos_tracker = ValueTracker(0)
        dash_updater = lambda m: m.move_to(path_start + pos_tracker.get_value() * (path_end - path_start))
        dash.add_updater(dash_updater)
        
        self.add(dash)
        self.play(pos_tracker.animate(run_time=3, rate_func=linear).set_value(1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # His average speed is exactly 25 meters per second.
        # Color: White (#FFFFFF)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF")
        
        avg_speed_label = Text("Average Speed = 25 m/s", font_size=24, color="#FFFFFF")
        # Issue 35 fix: Use D3-D5 area with scale_factor=0.8
        self.place_in_area(avg_speed_label, 'D3', 'D5', scale_factor=0.8)
        
        self.play(Write(avg_speed_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We freeze-frame his journey at exactly 2.5 seconds.
        # Color: Cyan (#00FFFF)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # Reset dash for the freeze-frame demonstration
        self.play(pos_tracker.animate(run_time=0.5).set_value(0))
        
        # 2.5 seconds out of 4 seconds = 0.625 of the total path
        freeze_point = 0.625
        
        # Vertical cyan line at the 2.5s position
        # Calculate horizontal position based on interpolation between C1 and C6
        freeze_x = self.grid["C1"][0] + freeze_point * (self.grid["C6"][0] - self.grid["C1"][0])
        freeze_line = Line(
            start=[freeze_x, self.grid["B1"][1], 0],
            end=[freeze_x, self.grid["D1"][1], 0],
            color="#00FFFF",
            stroke_width=4
        )
        
        self.play(pos_tracker.animate(run_time=1.875, rate_func=linear).set_value(freeze_point))
        self.play(Create(freeze_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Look closely at the position of his paws now.
        # Color: White (#FFFFFF)
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFFFF")
        
        # White circle to focus on "paws" (bottom of the circle)
        focus_circle = Circle(radius=0.2, color="#FFFFFF", stroke_width=4)
        # Position circle at the bottom edge of the cheetah
        focus_circle.move_to(dash.get_center() + 0.3 * DOWN)
        
        self.play(Create(focus_circle))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # How fast is he moving at this specific instant?
        # Color: Magenta (#FF00FF)
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF00FF")
        
        question_mark = Text("?", font_size=48, color="#FF00FF")
        # Issue 36 fix: Place at B4 with scale_factor=1.5
        self.place_at_grid(question_mark, 'B4', scale_factor=1.5) 
        
        # Flashing effect using renderer time
        def flashing_effect(m, dt):
            m.set_opacity(0.5 + 0.5 * np.sin(self.renderer.time * 10))

        self.add(question_mark)
        question_mark.add_updater(flashing_effect)
        
        self.play(FadeIn(question_mark))
        self.wait(3)
        
        # Clean up updaters
        dash.remove_updater(dash_updater)
        question_mark.remove_updater(flashing_effect)
