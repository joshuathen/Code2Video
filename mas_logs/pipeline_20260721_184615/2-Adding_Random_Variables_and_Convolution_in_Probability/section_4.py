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
            "The Convolution Formula: Flip and Slide", 
            [
                "The convolution formula mathematically defines this diagonal scan.",
                "First, we flip one distribution to reflect it.",
                "Next, we slide it across the other distribution.",
                "The overlap area at each step gives the sum.",
                "This 'flip and slide' creates the new distribution's shape."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Line 1 color: White (matches formula)
        self.lecture[0].set_color("#FFFFFF")
        
        formula = MathTex(
            r"(f * g)(z) = \int_{-\infty}^{\infty} f(x)g(z - x) \, dx",
            color="#FFFFFF"
        )
        # Issue 33: Position formula at A3-B6 with scale 0.7
        self.place_in_area(formula, "A3", "B6", scale_factor=0.7)
        
        self.play(Write(formula))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line 2 color: Green (matches f(x))
        self.lecture[1].set_color("#00FF00")

        # Issue 34: Position axes at C2-D6 with scale 0.9
        axes = NumberPlane(
            x_range=[-2, 3, 1],
            y_range=[0, 1.5, 1],
            x_length=5,
            y_length=2.5,
            axis_config={"include_tip": False}
        ).add_coordinates()
        self.place_in_area(axes, "C2", "D6", scale_factor=0.9)
        
        # Manim unit conversion for rectangles
        unit_w = axes.c2p(1,0)[0] - axes.c2p(0,0)[0]
        unit_h = axes.c2p(0,1)[1] - axes.c2p(0,0)[1]

        # f(x) is green square on [0, 1]
        f_square = Rectangle(
            width=unit_w, height=unit_h,
            fill_color="#00FF00", fill_opacity=0.5, 
            stroke_color="#00FF00"
        ).move_to(axes.c2p(0.5, 0.5))
        
        f_label = MathTex("f(x)", color="#00FF00").scale(0.7)
        f_label.next_to(f_square, UP, buff=0.1)

        # g(x) is magenta square on [0, 1] initially
        g_square = Rectangle(
            width=unit_w, height=unit_h,
            fill_color="#FF00FF", fill_opacity=0.5, 
            stroke_color="#FF00FF"
        ).move_to(axes.c2p(0.5, 0.5))
        
        g_label = MathTex("g(x)", color="#FF00FF").scale(0.7)
        g_label.next_to(g_square, UP, buff=0.1)

        self.play(Create(axes))
        self.play(FadeIn(f_square), FadeIn(f_label))
        self.wait(0.5)
        self.play(FadeIn(g_square), FadeIn(g_label))
        self.wait(0.5)
        
        # Flipping g(x) to g(-x)
        # We replace the label and move the square to [-1, 0]
        new_g_label = MathTex("g(-x)", color="#FF00FF").scale(0.7)
        new_g_label.move_to(axes.c2p(-0.5, 1.2))
        
        self.play(
            Rotate(g_square, axis=UP, angle=PI),
            g_square.animate.move_to(axes.c2p(-0.5, 0.5)),
            ReplacementTransform(g_label, new_g_label)
        )
        g_label = new_g_label
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Line 3 color: Magenta (matches sliding g(x))
        self.lecture[2].set_color("#FF00FF")

        # Sliding tracker for z. g(z-x) is at [z-1, z]
        z_tracker = ValueTracker(0.0) 
        
        def update_g(mob):
            z = z_tracker.get_value()
            mob.move_to(axes.c2p(z - 0.5, 0.5))
            
        def update_g_label(mob):
            mob.move_to(g_square.get_top() + UP * 0.2)

        g_square.add_updater(update_g)
        g_label.add_updater(update_g_label)

        # Slide into partial overlap (z=1 is full overlap for squares on [0,1])
        self.play(z_tracker.animate.set_value(1.0), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Line 4 color: Light Yellow (matches overlap)
        self.lecture[3].set_color("#FFFFE0")

        # Overlap rectangle highlight
        overlap_rect = Rectangle(
            width=unit_w, height=unit_h,
            fill_color="#FFFFE0", fill_opacity=0.8, stroke_width=0
        )
        
        def update_overlap(mob):
            z = z_tracker.get_value()
            left = max(0, z-1)
            right = min(1, z)
            if left < right:
                width = right - left
                center_x = (left + right) / 2
                mob.stretch_to_fit_width(width * unit_w)
                mob.stretch_to_fit_height(1 * unit_h)
                mob.move_to(axes.c2p(center_x, 0.5))
                mob.set_fill(opacity=0.8)
            else:
                mob.set_fill(opacity=0)

        overlap_rect.add_updater(update_overlap)
        self.add(overlap_rect)

        # Slide out of overlap
        self.play(z_tracker.animate.set_value(2.0), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Line 5 color: Cyan (matches triangle)
        self.lecture[4].set_color("#00FFFF")

        # Issue 35: Position result label at E4
        result_label = MathTex("(f*g)(z)", color="#00FFFF").scale(0.8)
        self.place_at_grid(result_label, "E4", scale_factor=0.8)

        # Result trace (triangle shape for convolution of two uniform rects)
        result_trace = VMobject(color="#00FFFF", stroke_width=4)
        result_trace.set_points_as_corners([axes.c2p(0, 0)])

        def update_trace(mob):
            z = z_tracker.get_value()
            if 0 <= z <= 1:
                val = z
            elif 1 < z <= 2:
                val = 2 - z
            else:
                val = 0
            
            new_point = axes.c2p(z, val)
            mob.add_points_as_corners([new_point])

        # Reset z and trace full convolution for final visualization
        z_tracker.set_value(0.0)
        
        self.add(result_trace)
        result_trace.add_updater(update_trace)

        self.play(
            z_tracker.animate.set_value(2.0),
            FadeIn(result_label),
            run_time=3.0,
            rate_func=linear
        )
        
        self.wait(2.0)

        # Cleanup updaters
        g_square.clear_updaters()
        g_label.clear_updaters()
        overlap_rect.clear_updaters()
        result_trace.clear_updaters()
