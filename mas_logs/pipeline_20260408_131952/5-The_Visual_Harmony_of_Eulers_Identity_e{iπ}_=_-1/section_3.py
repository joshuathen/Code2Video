from manim import *
import numpy as np

# Fix: Manim's config resolution uses .format() in a loop on paths containing curly braces.
# Escaping with "{{" is insufficient as the loop will strip them and crash on the next pass.
# We remove the curly braces from the input_file path to prevent the KeyError.
if "input_file" in config._d:
    config._d["input_file"] = str(config._d["input_file"]).replace("{", "").replace("}", "")

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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'The number e represents the limit of continuous growth.',
            'Real exponents grow outward along the number line.',
            'But an imaginary exponent changes the direction of growth.',
            'It forces growth to be perpendicular to our current position.',
            'This constant sideways push creates a perfect circular path.'
        ]
        self.setup_layout("Redefining e: Growth as a Movement", lecture_lines)

        # Colors
        C_REAL = "#33CCFF"   # Light Blue
        C_IMAG = "#FF66FF"   # Light Pink
        C_GOLD = "#FFD700"   # Gold
        C_AXES = "#888888"   # Gray

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create Complex Plane centered in the grid area
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": C_AXES, "include_tip": True}
        )
        self.place_in_area(axes, "A1", "F6")
        
        # Labels for axes
        labels = axes.get_axis_labels(
            x_label=Text("Re", font_size=20), 
            y_label=Text("Im", font_size=20)
        )
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(C_REAL)
        )
        
        # Point at (1, 0)
        start_point = axes.c2p(1, 0)
        dot = Dot(start_point, color=WHITE)
        
        # Growth vector (Velocity) - pointing right
        vel_vector = Arrow(
            start=axes.c2p(1, 0),
            end=axes.c2p(2, 0),
            buff=0,
            color=C_REAL
        )
        vel_label = Text("e^x growth", color=C_REAL, font_size=20)
        # Resolved Issue 38 & 39: Positioning label appropriately for real growth
        self.place_at_grid(vel_label, "C6", scale_factor=0.8)
        
        self.play(FadeIn(dot), GrowArrow(vel_vector))
        self.play(Write(vel_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(C_IMAG)
        )
        
        # Rotate the velocity vector 90 degrees (Imaginary growth)
        new_vel_vector = Arrow(
            start=axes.c2p(1, 0),
            end=axes.c2p(1, 1),
            buff=0,
            color=C_IMAG
        )
        imag_label = Text("e^ix growth", color=C_IMAG, font_size=20)
        # Resolved Issue 38 & 40: Positioning label appropriately for imaginary growth
        self.place_at_grid(imag_label, "B4", scale_factor=0.8)
        
        self.play(
            Transform(vel_vector, new_vel_vector),
            ReplacementTransform(vel_label, imag_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(C_IMAG)
        )
        
        radius_line = Line(axes.c2p(0, 0), axes.c2p(1, 0), color=WHITE, stroke_width=2)
        perp_symbol = RightAngle(radius_line, vel_vector, length=0.2, color=WHITE)
        
        self.play(Create(radius_line))
        self.play(Create(perp_symbol))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(C_GOLD)
        )
        
        angle_tracker = ValueTracker(0)
        
        def get_pos(angle):
            return axes.c2p(np.cos(angle), np.sin(angle))
            
        def get_tangent(angle):
            p1 = get_pos(angle)
            p2 = axes.c2p(np.cos(angle) - 0.8*np.sin(angle), np.sin(angle) + 0.8*np.cos(angle))
            return [p1, p2]

        path = TracedPath(dot.get_center, stroke_color=C_GOLD, stroke_width=4)
        self.add(path)

        dot.add_updater(lambda d: d.move_to(get_pos(angle_tracker.get_value())))
        
        self.remove(vel_vector, perp_symbol)
        dynamic_vel = Arrow(color=C_IMAG, buff=0)
        dynamic_vel.add_updater(lambda m: m.put_start_and_end_on(*get_tangent(angle_tracker.get_value())))
        
        radius_line.add_updater(lambda l: l.put_start_and_end_on(axes.c2p(0,0), get_pos(angle_tracker.get_value())))
        
        self.add(dynamic_vel)
        self.play(FadeOut(imag_label))

        self.play(angle_tracker.animate.set_value(TAU), run_time=5, rate_func=linear)
        
        dot.clear_updaters()
        dynamic_vel.clear_updaters()
        radius_line.clear_updaters()
        
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
