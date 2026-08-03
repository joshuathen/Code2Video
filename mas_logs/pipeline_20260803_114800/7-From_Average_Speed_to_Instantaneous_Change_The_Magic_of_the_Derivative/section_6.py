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
        # Data from storyboard
        title = "Application: The Rocket Launch"
        lecture_lines = [
            "Derivatives calculate precise changes in real-time.",
            "This rocket's speed changes at every single moment.",
            "The derivative gives the exact velocity needed."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        PATH_COLOR = "#FFFFFF"
        TANGENT_COLOR = "#FFA500"
        ROCKET_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Define the parabolic path: y = 0.12(x-0.5)^2 - 1.8
        # Domain x in [0.5, 5.5]
        path = ParametricFunction(
            lambda t: np.array([
                t, 
                0.12 * (t - 0.5)**2 - 1.8, 
                0
            ]),
            t_range=[0.5, 5.5],
            color=PATH_COLOR
        )
        
        # Rocket asset integration (Issue 21)
        rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg")
        rocket.set_color(ROCKET_COLOR)
        rocket.scale(0.3)
        rocket.curr_angle = PI/2 # SVG rockets usually point UP

        # Setup position tracker
        t_tracker = ValueTracker(0) # proportion from 0 to 1
        
        def get_path_data(p):
            # p is proportion in [0, 1]
            # Domain mapping: x = 0.5 + 5 * p
            x = 0.5 + 5 * p
            pos = path.point_from_proportion(p)
            # dy/dx = 0.24 * (x - 0.5)
            slope = 0.24 * (x - 0.5)
            angle = np.arctan2(slope, 1)
            return pos, angle, slope

        # Initial setup
        pos, angle, slope = get_path_data(0)
        rocket.move_to(pos)
        rocket.rotate(angle - rocket.curr_angle)
        rocket.curr_angle = angle

        self.play(Create(path), run_time=2)
        self.play(FadeIn(rocket))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Tangent line representing velocity
        tangent_line = Line(LEFT, RIGHT, color=TANGENT_COLOR, stroke_width=4).scale(0.6)
        
        # Updaters for rocket and tangent
        def update_rocket(m):
            p = t_tracker.get_value()
            pos, angle, _ = get_path_data(p)
            m.move_to(pos)
            m.rotate(angle - m.curr_angle)
            m.curr_angle = angle

        def update_tangent(m):
            p = t_tracker.get_value()
            pos, angle, _ = get_path_data(p)
            m.move_to(pos)
            # Line is initially horizontal (angle 0)
            # We want it at 'angle'
            # We must be careful not to accumulate rotations.
            # Easiest: rotate a fresh line each time or use a property
            curr_a = getattr(m, "curr_a", 0)
            m.rotate(angle - curr_a)
            m.curr_a = angle

        rocket.add_updater(update_rocket)
        tangent_line.add_updater(update_tangent)
        
        self.play(FadeIn(tangent_line))
        # Move along part of the path
        self.play(t_tracker.animate.set_value(0.4), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Slope value indicator
        slope_label = MathTex(r"v = ", color=TANGENT_COLOR)
        # Fix overlap (Issue 29): Move from A5 to A1
        self.place_at_grid(slope_label, "A1", scale_factor=0.8)
        
        slope_value = DecimalNumber(0, color=TANGENT_COLOR, num_decimal_places=2)
        slope_value.next_to(slope_label, RIGHT)
        
        # Update function for the decimal number
        def update_slope_val(m):
            p = t_tracker.get_value()
            _, _, slope = get_path_data(p)
            m.set_value(slope)
            
        slope_value.add_updater(update_slope_val)
        
        self.play(FadeIn(slope_label), FadeIn(slope_value))
        
        # Finish ascent
        self.play(t_tracker.animate.set_value(1.0), run_time=3, rate_func=smooth)
        self.wait(3)
        
        # Clean up
        rocket.clear_updaters()
        tangent_line.clear_updaters()
        slope_value.clear_updaters()
