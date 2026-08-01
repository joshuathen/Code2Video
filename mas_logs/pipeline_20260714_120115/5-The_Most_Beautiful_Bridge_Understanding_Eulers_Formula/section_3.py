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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from shared state - Synchronized to 3 lines (Issue 40)
        title_text = "The Engine: Continuous Growth in a New Direction"
        lecture_lines = [
            "The number e typically represents continuous exponential growth.",
            "Multiplying by i rotates this growth by ninety degrees.",
            "This constant turning forces the path into a perfect circle."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors for lecture/animation mapping
        c1, c2, c3 = YELLOW, BLUE, WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(c1)
        
        # Issue 32: Plane positioned B2-F6 to allow Row A for header labels
        plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        ).set_z_index(-1)
        self.place_in_area(plane, "B2", "F6")
        
        # Issue 23: Load engine icon asset
        engine = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/engine.svg")
        engine.scale(0.3).move_to(plane.n2p(1 + 0j))
        
        # Issue 30: e_label at A4, scale 0.7. No MathTex (Issue 40).
        e_label = Text("e", color=c1, slant=ITALIC)
        self.place_at_grid(e_label, "A4", scale_factor=0.7)
        
        # Velocity vector representing real growth (forward)
        velocity = Arrow(
            start=plane.n2p(1),
            end=plane.n2p(1.5),
            color=c1,
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.2
        )
        
        self.play(
            FadeIn(plane), 
            Write(e_label), 
            FadeIn(engine), 
            GrowArrow(velocity)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(c2)
        
        # Issue 31: i_label at A5, scale 0.7. No MathTex (Issue 40).
        i_label = Text("i", color=c2, slant=ITALIC)
        self.place_at_grid(i_label, "A5", scale_factor=0.7)
        
        # Rotate velocity vector 90 degrees CCW
        tangent_velocity_end = plane.n2p(1 + 0.5j)
        
        self.play(
            Write(i_label),
            velocity.animate.put_start_and_end_on(plane.n2p(1), tangent_velocity_end).set_color(c2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(c3)
        
        # Continuous turning into a circle
        angle_tracker = ValueTracker(0)
        
        # Add updaters for engine and arrow to maintain circular path and tangency
        # Note: Do not use .become() inside updaters for performance (Lesson L014)
        engine.add_updater(lambda m: m.move_to(plane.n2p(np.exp(1j * angle_tracker.get_value()))))
        
        def update_velocity(v):
            theta = angle_tracker.get_value()
            pos = np.exp(1j * theta)
            # Tangent direction is i * pos = exp(i * (theta + pi/2))
            direction = np.exp(1j * (theta + np.pi/2))
            v.put_start_and_end_on(
                plane.n2p(pos),
                plane.n2p(pos + 0.5 * direction)
            )
        
        velocity.add_updater(update_velocity)
        
        # Trace the path
        path = TracedPath(engine.get_center, stroke_color=WHITE, stroke_width=4)
        self.add(path)
        
        # Complete the circular orbit
        self.play(angle_tracker.animate.set_value(TAU), run_time=5, rate_func=linear)
        self.wait(2)
