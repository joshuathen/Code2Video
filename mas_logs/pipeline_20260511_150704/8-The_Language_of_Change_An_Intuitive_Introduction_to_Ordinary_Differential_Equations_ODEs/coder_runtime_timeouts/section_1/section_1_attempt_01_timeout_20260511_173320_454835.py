from manim import *
import numpy as np

# Base class for maintaining layout and positioning consistency
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
        # Initial Teaching Scene Setup
        lecture_lines_text = [
            "Numbers alone cannot describe a world in motion.",
            "We need a way to capture how things change.",
            "Differential equations are rules for this growth."
        ]
        self.setup_layout("The Hook: Capturing Movement", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        # Description: Create a simple cheetah silhouette and a ground line. [Color: #FFFF00]
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Ground line spanning the right side
        ground = Line(
            start=self.grid["F1"] + LEFT*0.5, 
            end=self.grid["F6"] + RIGHT*0.5, 
            color=GREY_B
        )
        
        # Construct a simple cheetah silhouette from shapes
        body = RoundedRectangle(width=0.8, height=0.4, corner_radius=0.1, fill_opacity=1, color="#FFFF00")
        head = Circle(radius=0.15, fill_opacity=1, color="#FFFF00").next_to(body, RIGHT, buff=-0.1, aligned_edge=UP)
        leg_front = Line(ORIGIN, DOWN*0.25, color="#FFFF00", stroke_width=4).next_to(body, DOWN, buff=0, aligned_edge=LEFT).shift(RIGHT*0.1)
        leg_back = Line(ORIGIN, DOWN*0.25, color="#FFFF00", stroke_width=4).next_to(body, DOWN, buff=0, aligned_edge=RIGHT).shift(LEFT*0.1)
        cheetah = VGroup(body, head, leg_front, leg_back)
        
        # Position cheetah at rest on the ground
        self.place_at_grid(cheetah, "F1", scale_factor=0.8)
        cheetah.shift(UP * 0.2) 
        
        self.play(Create(ground))
        self.play(FadeIn(cheetah))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Move the cheetah across with increasing speed and a velocity vector. [Color: #00FF00]
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Physics simulation using ValueTracker for constant acceleration
        time_tracker = ValueTracker(0)
        accel_const = 1.2
        start_x_pos = cheetah.get_x()
        
        # Velocity vector and acceleration value label
        vel_vector = Arrow(ORIGIN, RIGHT, color="#00FF00", buff=0, stroke_width=5)
        accel_label = Text(f"a = {accel_const}", font_size=16, color="#00FF00")
        
        # Updaters for smooth motion and vector adjustment
        def update_cheetah_pos(mob):
            t = time_tracker.get_value()
            # Displacement = 0.5 * a * t^2
            mob.set_x(start_x_pos + 0.5 * accel_const * (t**2))

        def update_velocity_vector(mob):
            t = time_tracker.get_value()
            # velocity = a * t
            v_val = accel_const * t
            mob.set_width(max(0.1, v_val * 0.6), stretch=True, about_edge=LEFT)
            mob.next_to(cheetah, UP, buff=0.1)
            mob.set_x(cheetah.get_x())

        def update_accel_tag(mob):
            mob.next_to(cheetah, DOWN, buff=0.1)
            mob.set_x(cheetah.get_x())

        cheetah.add_updater(update_cheetah_pos)
        vel_vector.add_updater(update_velocity_vector)
        accel_label.add_updater(update_accel_tag)
        
        self.add(vel_vector, accel_label)
        
        # Perform acceleration animation
        self.play(time_tracker.animate.set_value(2.5), run_time=3, rate_func=linear)
        
        # Freeze current state
        cheetah.clear_updaters()
        vel_vector.clear_updaters()
        accel_label.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Display text 'Rate of Change' and link it to acceleration. [Color: #FFFFFF]
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Dynamic appearance of 'Rate of Change' text
        roc_text = Text("Rate of Change", font_size=24, color="#FFFFFF")
        self.place_at_grid(roc_text, "B3", scale_factor=1.0)
        
        # Formula indicating the 'Rule for growth' / Law of motion
        diff_eq_formula = MathTex(r"\frac{dv}{dt} = a", color="#FFFFFF")
        self.place_at_grid(diff_eq_formula, "C3", scale_factor=1.2)
        
        # Visual link between text and formula
        connection = Arrow(roc_text.get_bottom(), diff_eq_formula.get_top(), color=WHITE, buff=0.2)
        
        self.play(Write(roc_text))
        self.play(Create(connection), FadeIn(diff_eq_formula))
        
        self.wait(2)
