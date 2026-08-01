from manim import *

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
            "The Exponential Path: Compounding Rotations", 
            [
                'Start with a unit vector pointing towards one.', 
                'The imaginary unit i forces a perpendicular growth rate.', 
                'Small compounding steps push the vector sideways.', 
                'This continuous steering creates a perfect circular path.', 
                'We define this as continuous perpendicular growth.'
            ]
        )
        
        # Colors
        COLOR_RADIUS = "#FFFFFF"
        COLOR_GROWTH = "#00FFFF"
        
        # Center point for the circle
        center_point = self.grid["D3"]
        unit_length = 1.5

        # === Animation for Lecture Line 1 ===
        # Show a white (#FFFFFF) vector at (1,0) with a small 'Growth' label in cyan (#00FFFF).
        self.lecture[0].set_color(COLOR_RADIUS)
        radius_vec = Vector([unit_length, 0, 0], color=COLOR_RADIUS)
        radius_vec.shift(center_point)
        
        growth_lbl = Text("Growth", font_size=18, color=COLOR_GROWTH)
        self.place_at_grid(growth_lbl, "B4", scale_factor=0.8)
        
        self.play(Create(radius_vec), Write(growth_lbl))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Add a cyan (#00FFFF) velocity vector at the tip of the main vector, pointing exactly perpendicular (upwards).
        self.lecture[1].set_color(COLOR_GROWTH)
        vel_vec = Arrow(
            start=radius_vec.get_end(), 
            end=radius_vec.get_end() + UP * 0.8, 
            color=COLOR_GROWTH,
            buff=0
        )
        
        self.play(GrowArrow(vel_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the vector tip tracing an arc while the cyan (#00FFFF) velocity vector continuously rotates to stay perpendicular.
        self.lecture[2].set_color(COLOR_GROWTH)
        
        theta_tracker = ValueTracker(0)
        
        def update_radius(m):
            angle = theta_tracker.get_value()
            new_end = center_point + np.array([np.cos(angle), np.sin(angle), 0]) * unit_length
            m.put_start_and_end_on(center_point, new_end)

        def update_vel(m):
            angle = theta_tracker.get_value()
            tip_pos = center_point + np.array([np.cos(angle), np.sin(angle), 0]) * unit_length
            perp_dir = np.array([-np.sin(angle), np.cos(angle), 0])
            m.put_start_and_end_on(tip_pos, tip_pos + perp_dir * 0.8)

        # Trace the path
        path = TracedPath(radius_vec.get_end, stroke_color=COLOR_GROWTH, stroke_width=4)
        self.add(path)
        
        radius_vec.add_updater(update_radius)
        vel_vec.add_updater(update_vel)
        
        self.play(theta_tracker.animate.set_value(PI / 2), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Show a sequence of 15 tiny linear segments in cyan (#00FFFF) approximating the curve of the circle.
        self.lecture[3].set_color(COLOR_GROWTH)
        
        segments = VGroup()
        start_angle = PI / 2
        end_angle = 3 * PI / 2
        num_segments = 15
        d_theta = (end_angle - start_angle) / num_segments
        
        for i in range(num_segments):
            a1 = start_angle + i * d_theta
            a2 = a1 + d_theta
            p1 = center_point + np.array([np.cos(a1), np.sin(a1), 0]) * unit_length
            p2 = center_point + np.array([np.cos(a2), np.sin(a2), 0]) * unit_length
            line = Line(p1, p2, color=COLOR_GROWTH, stroke_width=4)
            segments.add(line)

        # Hide smooth path and show segments instead
        self.play(FadeOut(path), run_time=0.3)
        
        self.play(
            Succession(*[
                AnimationGroup(
                    theta_tracker.animate(run_time=0.15).set_value(start_angle + (i+1)*d_theta),
                    Create(segments[i], run_time=0.15)
                ) for i in range(num_segments)
            ]),
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Smooth the segments into a perfect, solid white (#FFFFFF) unit circle.
        self.lecture[4].set_color(COLOR_RADIUS)
        
        full_circle = Circle(radius=unit_length, color=COLOR_RADIUS).move_to(center_point)
        
        self.play(
            FadeOut(growth_lbl),
            FadeOut(vel_vec),
            FadeOut(segments),
            theta_tracker.animate.set_value(2 * PI),
            Create(full_circle),
            run_time=1.5
        )
        
        # Label the circle correctly based on feedback (Issue 48/49)
        label = Text("Continuous Perpendicular Growth", font_size=20, color=COLOR_RADIUS)
        self.place_in_area(label, 'B2', 'B5', scale_factor=0.7)
        
        self.play(Write(label))
        self.wait(2)

        # Clean up updaters
        radius_vec.remove_updater(update_radius)
        vel_vec.remove_updater(update_vel)
