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
        # Data from storyboard
        title_text = "Visualizing Convolution: Flip and Slide"
        lecture_lines = [
            "First, we flip the second distribution's graph horizontally.",
            "Then, we slide it along the x-axis by distance z.",
            "At each position, multiply the two overlapping functions.",
            "The area under this product is the resulting density.",
            "Uniform distributions convolve into a triangular shape."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_X = "#ADD8E6"
        COLOR_Y = "#90EE90"
        COLOR_OVERLAP = "#FFFF00"
        COLOR_RESULT = "#FF69B4"
        COLOR_FLASH = "#FFFFFF"

        # 1. Setup Axes
        axes = Axes(
            x_range=[-1.5, 2.5, 1],
            y_range=[0, 1.5, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": True, "font_size": 20, "color": GREY_C}
        ).add_coordinates()
        self.place_in_area(axes, "B1", "E6", scale_factor=1.0)
        
        x_label = axes.get_x_axis_label("t").scale(0.7)
        y_label = axes.get_y_axis_label("f(t)").scale(0.7)

        # f_X(t) - fixed square wave (Uniform[0,1])
        fx_points = [axes.c2p(-1.5, 0), axes.c2p(0, 0), axes.c2p(0, 1), axes.c2p(1, 1), axes.c2p(1, 0), axes.c2p(2.5, 0)]
        fx_graph = VMobject(color=COLOR_X, stroke_width=4).set_points_as_corners(fx_points)
        fx_text = MathTex("f_X(t)", color=COLOR_X, font_size=24)
        # Fix Issue 49: Move 'fx_text' to 'A3' (scale 0.8)
        self.place_at_grid(fx_text, 'A3', scale_factor=0.8)

        # f_Y(t) - initial state (Uniform[0,1])
        fy_init_points = [axes.c2p(-1.5, 0), axes.c2p(0, 0), axes.c2p(0, 1), axes.c2p(1, 1), axes.c2p(1, 0), axes.c2p(2.5, 0)]
        fy_graph = VMobject(color=COLOR_Y, stroke_width=4).set_points_as_corners(fy_init_points)
        fy_text = MathTex("f_Y(t)", color=COLOR_Y, font_size=24)
        # Fix Issue 49: Move initial 'fy_text' to 'A5' (scale 0.8)
        self.place_at_grid(fy_text, 'A5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # First, we flip the second distribution's graph horizontally.
        self.lecture[0].set_color(COLOR_Y)
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(fx_graph), Write(fx_text))
        self.play(Create(fy_graph), Write(fy_text))
        self.wait(1)

        # Flipping f_Y: (t) -> (-t) which moves the pulse from [0,1] to [-1, 0]
        fy_flipped_points = [axes.c2p(-1.5, 0), axes.c2p(-1, 0), axes.c2p(-1, 1), axes.c2p(0, 1), axes.c2p(0, 0), axes.c2p(2.5, 0)]
        fy_flipped_text = MathTex("f_Y(-t)", color=COLOR_Y, font_size=24)
        # Fix Issue 49: Move 'fy_flipped_text' to 'A1' (scale 0.8)
        self.place_at_grid(fy_flipped_text, 'A1', scale_factor=0.8)

        self.play(
            fy_graph.animate.set_points_as_corners(fy_flipped_points),
            Transform(fy_text, fy_flipped_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Then, we slide it along the x-axis by distance z.
        self.lecture[1].set_color(COLOR_Y)
        
        z_tracker = ValueTracker(0.0) 
        
        def update_fy_sliding(mob):
            z = z_tracker.get_value()
            pts = [axes.c2p(-1.5, 0), axes.c2p(z-1, 0), axes.c2p(z-1, 1), axes.c2p(z, 1), axes.c2p(z, 0), axes.c2p(2.5, 0)]
            mob.set_points_as_corners(pts)

        fy_graph.add_updater(update_fy_sliding)
        
        fy_label_moving = MathTex("f_Y(z-t)", color=COLOR_Y, font_size=24)
        def update_fy_label(mob):
            z = z_tracker.get_value()
            mob.move_to(axes.c2p(z-0.5, 1.3, 0))
        
        fy_label_moving.add_updater(update_fy_label)
        
        self.play(
            FadeOut(fy_text),
            FadeIn(fy_label_moving),
            z_tracker.animate.set_value(0.1), 
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # At each position, multiply the two overlapping functions.
        self.lecture[2].set_color(COLOR_OVERLAP)
        
        overlap_poly = VMobject(fill_color=COLOR_OVERLAP, fill_opacity=0.5, stroke_width=0)
        
        def update_overlap(mob):
            z = z_tracker.get_value()
            start = max(0, z-1)
            end = min(1, z)
            if start < end:
                mob.set_points_as_corners([
                    axes.c2p(start, 0),
                    axes.c2p(start, 1),
                    axes.c2p(end, 1),
                    axes.c2p(end, 0),
                    axes.c2p(start, 0)
                ])
                mob.set_fill(opacity=0.5)
            else:
                mob.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])
                mob.set_fill(opacity=0)

        overlap_poly.add_updater(update_overlap)
        self.add(overlap_poly)
        
        self.play(z_tracker.animate.set_value(0.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The area under this product is the resulting density.
        self.lecture[3].set_color(COLOR_RESULT)
        
        result_curve = VMobject(color=COLOR_RESULT, stroke_width=5)
        result_curve.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])

        def update_result(mob):
            z = z_tracker.get_value()
            steps = np.linspace(0, z, 100)
            pts = []
            for s in steps:
                val = 0
                if 0 <= s <= 1: val = s
                elif 1 < s <= 2: val = 2 - s
                elif s > 2: val = 0
                pts.append(axes.c2p(s, val))
            if len(pts) > 1:
                mob.set_points_as_corners(pts)
            
        result_curve.add_updater(update_result)
        self.add(result_curve)

        self.play(z_tracker.animate.set_value(1.0), run_time=3)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Uniform distributions convolve into a triangular shape.
        self.lecture[4].set_color(COLOR_RESULT)
        
        flash = Flash(axes.c2p(1, 1), color=COLOR_FLASH, flash_radius=0.4, line_length=0.2)
        self.play(flash)
        
        self.play(z_tracker.animate.set_value(2.2), run_time=3)
        self.wait(2)
