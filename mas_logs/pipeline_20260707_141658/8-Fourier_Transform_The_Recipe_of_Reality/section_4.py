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
        self.setup_layout(
            "The Mechanism: The Winding Machine",
            [
                "We can wrap a signal around a circular path.",
                "Most winding speeds create a balanced, centered shape.",
                "When speeds match, the shape becomes unbalanced.",
                "The center of mass shifts away from origin.",
                "This peak identifies a frequency within the signal."
            ]
        )

        # Right side center - Middle of the right grid area
        center_pos = np.array([3.0, -0.3, 0])
        
        # Constants
        SIGNAL_FREQ = 1.0
        T_MAX = 4.0
        
        # Value Tracker for the winding frequency
        wind_freq_tracker = ValueTracker(0.4)
        
        # Helper function to get point on the wound path
        def get_wound_point(t, wind_freq):
            # Radius varies as s(t) = 1.2 + 0.6 * cos(2 * pi * f_sig * t)
            radius = 1.2 + 0.6 * np.cos(2 * np.pi * SIGNAL_FREQ * t)
            angle = -2 * np.pi * wind_freq * t
            return center_pos + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

        # === Animation for Lecture Line 1 ===
        # "We can wrap a signal around a circular path."
        self.lecture[0].set_color(YELLOW)
        
        origin_dot = Dot(center_pos, color="#FFFFFF")
        self.add(origin_dot)
        
        # Initial wound shape as a ParametricFunction for smooth introduction
        wound_shape = ParametricFunction(
            lambda t: get_wound_point(t, wind_freq_tracker.get_value()),
            t_range=[0, T_MAX],
            color="#00FF00"
        )
        
        self.play(Create(wound_shape), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Most winding speeds create a balanced, centered shape."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "When speeds match, the shape becomes unbalanced."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Center of mass calculation
        def get_cm():
            pts = [get_wound_point(t, wind_freq_tracker.get_value()) for t in np.linspace(0, T_MAX, 100)]
            return np.mean(pts, axis=0)

        cm_dot = Dot(get_cm(), color="#FF0000")
        cm_label = Text("Center of Mass", font_size=16, color="#FF0000")
        
        # Position label in area F4-F6 (bottom right) to avoid overlap (Issue 29/30)
        self.place_in_area(cm_label, 'F4', 'F6', scale_factor=0.6)
        
        # Optimized updaters
        def update_wound_shape(m):
            # Efficiently update points instead of recreating the object
            pts = [get_wound_point(t, wind_freq_tracker.get_value()) for t in np.linspace(0, T_MAX, 150)]
            m.set_points_smoothly(pts)
            
        wound_shape.add_updater(update_wound_shape)
        cm_dot.add_updater(lambda m: m.move_to(get_cm()))

        self.play(FadeIn(cm_dot), Write(cm_label))
        self.wait(1)
        
        # Animate frequency change to 1.0 (matching SIGNAL_FREQ)
        self.play(
            wind_freq_tracker.animate.set_value(1.0),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The center of mass shifts away from origin."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Visualizing the shift with an arrow
        arrow = Arrow(origin_dot.get_center(), cm_dot.get_center(), color="#FF0000", buff=0.1)
        # Keep arrow persistent and updated
        arrow.add_updater(lambda m: m.put_start_and_end_on(origin_dot.get_center(), cm_dot.get_center()))
        
        self.play(Create(arrow))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # "This peak identifies a frequency within the signal."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.wait(3)
        
        # Cleanup updaters
        wound_shape.clear_updaters()
        cm_dot.clear_updaters()
        arrow.clear_updaters()
