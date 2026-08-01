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
        plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, "A1", "F6")
        
        theta_tracker = ValueTracker(0)
        
        # Growth Vector (Position in complex plane)
        radius_vec = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(1, 0),
            buff=0,
            color=WHITE,
            stroke_width=4
        )
        
        def update_radius_vec(v):
            angle = theta_tracker.get_value()
            v.put_start_and_end_on(
                plane.c2p(0, 0),
                plane.c2p(np.cos(angle), np.sin(angle))
            )
        
        radius_vec.add_updater(update_radius_vec)
        
        # Sideways Push Vector (Derivative/Force)
        push_vec = Arrow(
            start=plane.c2p(1, 0),
            end=plane.c2p(1, 0.5),
            buff=0,
            color=YELLOW,
            stroke_width=3
        )
        push_vec.set_opacity(0)
        
        def update_push_vec(v):
            angle = theta_tracker.get_value()
            pos = np.array([np.cos(angle), np.sin(angle), 0])
            # Perpendicular vector (rotate 90 deg: -y, x)
            perp = np.array([-np.sin(angle), np.cos(angle), 0]) * 0.7
            v.put_start_and_end_on(
                plane.c2p(pos[0], pos[1]),
                plane.c2p(pos[0] + perp[0], pos[1] + perp[1])
            )
            
        push_vec.add_updater(update_push_vec)

        # Traced Circle Path
        path = TracedPath(
            radius_vec.get_end,
            stroke_color="#FF5555",
            stroke_width=5
        )

        # Labels
        label_circle = Text("Circular Motion", font_size=20, color=WHITE)
        self.place_at_grid(label_circle, "A5")
        label_circle.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Description: Show a vector (#FFFFFF) pointing from the origin to 1 on the real axis
        self.lecture[0].set_opacity(1.0).set_color(WHITE)
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(radius_vec), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Introduce 'i' into the exponent; the growth vector turns 90 degrees 
        self.lecture[0].set_opacity(0.3)
        self.lecture[1].set_opacity(1.0).set_color(YELLOW)
        
        self.play(push_vec.animate.set_opacity(1), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: As the exponent grows, the vector's tip traces a smooth circular path (#FF5555)
        self.lecture[1].set_opacity(0.3)
        self.lecture[2].set_opacity(1.0).set_color("#FF5555")
        
        self.add(path)
        self.play(theta_tracker.animate.set_value(TAU), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Description: Show a small perpendicular force arrow (#5555FF) at the tip
        self.lecture[2].set_opacity(0.3)
        self.lecture[3].set_opacity(1.0).set_color("#5555FF")
        
        # We update the push_vec color to blue
        push_vec.clear_updaters() # Pause to change color style
        push_vec.set_color("#5555FF")
        push_vec.add_updater(update_push_vec)
        
        # Rotate a bit more to emphasize the blue push
        self.play(theta_tracker.animate.set_value(TAU + PI/2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Description: Label the resulting path as 'Circular Motion' (#FFFFFF)
        self.lecture[3].set_opacity(0.3)
        self.lecture[4].set_opacity(1.0).set_color(WHITE)
        
        self.play(Write(label_circle), run_time=1)
        self.wait(2)
