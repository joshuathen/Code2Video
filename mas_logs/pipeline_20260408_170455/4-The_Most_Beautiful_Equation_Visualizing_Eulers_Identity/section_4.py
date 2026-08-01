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
        # 1. Setup Layout
        lecture_lines = [
            'What happens if our growth rate is imaginary?',
            'The imaginary unit i pushes growth sideways.',
            'Instead of growing outward, the path begins to curve.',
            'This constant perpendicular push forms a perfect circle.',
            'It acts like a steering wheel for growth.'
        ]
        self.setup_layout("The Sideways Push (The i-Exponent)", lecture_lines)
        
        # Set initial low opacity for all lines
        for line in self.lecture:
            line.set_opacity(0.3)

        # 2. Right Side Visualization Assets
        # Complex Plane centered in the right half
        plane = ComplexPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, "A1", "F6")
        
        theta_tracker = ValueTracker(0)
        
        # Tracker dot for the path (more efficient than tracking the arrow tip directly)
        dot_tip = Dot(radius=0).set_opacity(0)
        def update_dot(d):
            t = theta_tracker.get_value()
            d.move_to(plane.c2p(np.cos(t), np.sin(t)))
        dot_tip.add_updater(update_dot)
        self.add(dot_tip)

        # Position Vector (Position in complex plane)
        pos_vec = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(1, 0),
            buff=0,
            color=WHITE,
            stroke_width=4
        )
        def update_pos_vec(v):
            v.put_start_and_end_on(plane.c2p(0, 0), dot_tip.get_center())
        
        # Push Vector (Growth direction / Velocity)
        push_vec = Arrow(
            color=YELLOW,
            buff=0,
            stroke_width=4
        )
        def update_push_vec(v):
            t = theta_tracker.get_value()
            curr_pos = dot_tip.get_center()
            # Direction in complex units: i * e^{it} = -sin(t) + i*cos(t)
            dir_point = plane.c2p(-np.sin(t), np.cos(t))
            origin_point = plane.c2p(0, 0)
            v_dir = (dir_point - origin_point) * 0.7 # Scale vector size
            v.put_start_and_end_on(curr_pos, curr_pos + v_dir)

        # Traced Circle Path
        path = TracedPath(
            dot_tip.get_center,
            stroke_color="#FF5555",
            stroke_width=5
        )

        # Labels
        label_circle = Text("Circular Motion", font_size=22, color=WHITE)
        # [Fix Issue 47] Positioning improved to area A5-A6
        self.place_in_area(label_circle, "A5", "A6", scale_factor=0.8)
        label_circle.set_opacity(0)

        # Formula label as an additional visual aid
        formula_e_it = Text("e^it", font_size=24, color=YELLOW)
        # [Fix Issue 46] Positioning moved further right to A2-A3 and uses area-based placement
        self.place_in_area(formula_e_it, "A2", "A3", scale_factor=0.8)
        formula_e_it.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Description: Show a vector (#FFFFFF) pointing from the origin to 1 on the real axis
        self.lecture[0].set_opacity(1.0).set_color(WHITE)
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(pos_vec), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Introduce 'i' into the exponent; the growth vector turns 90 degrees 
        self.lecture[0].set_opacity(0.3)
        self.lecture[1].set_opacity(1.0).set_color(YELLOW)
        
        push_vec.add_updater(update_push_vec)
        self.play(
            FadeIn(push_vec),
            formula_e_it.animate.set_opacity(1),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: As the exponent grows, the vector's tip traces a smooth circular path (#FF5555)
        self.lecture[1].set_opacity(0.3)
        self.lecture[2].set_opacity(1.0).set_color("#FF5555")
        
        pos_vec.add_updater(update_pos_vec)
        self.add(path)
        # First curve to show initial turning motion
        self.play(theta_tracker.animate.set_value(PI/2), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Description: Show a small perpendicular force arrow (#5555FF) at the tip
        self.lecture[2].set_opacity(0.3)
        self.lecture[3].set_opacity(1.0).set_color("#5555FF")
        
        # Change the push vector color to blue to emphasize the concept change
        push_vec.set_color("#5555FF")
        # Complete the circular rotation
        self.play(theta_tracker.animate.set_value(TAU), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Description: Label the resulting path as 'Circular Motion' (#FFFFFF)
        self.lecture[3].set_opacity(0.3)
        self.lecture[4].set_opacity(1.0).set_color(WHITE)
        
        self.play(label_circle.animate.set_opacity(1), run_time=1)
        self.wait(3)
