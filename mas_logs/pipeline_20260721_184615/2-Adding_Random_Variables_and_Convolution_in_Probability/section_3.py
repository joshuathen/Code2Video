from manim import *
import numpy as np

# [L002] Scaling labels to 0.7-0.8 and following Proximity Rule.
# [L008] Hexadecimal color strings.
# [L004] Correct 'Indicate' class.
# [L017] 'axes.input_to_graph_point' for coordinate calculation.
# [L024] 'rate_functions.' prefix.

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
        # Section data
        title = "The Geometric Intuition of Z = X + Y"
        lines = [
            "Let the sum Z equal a constant value z.",
            "This equation forms a diagonal line on the plane.",
            "To find P(Z=z), we scan this specific line.",
            "We accumulate joint probabilities along this diagonal path.",
            "This scan explains how the sum's distribution emerges."
        ]
        self.setup_layout(title, lines)

        # Colors (L008: Hex codes)
        COLOR_Z = "#FFFFFF"
        COLOR_LINE = "#FFFF00"
        COLOR_PULSE = "#FF4500"
        COLOR_BARS = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        # Let the sum Z equal a constant value z.
        self.lecture[0].set_color(COLOR_Z)
        
        eq_z = MathTex("Z = X + Y = z", color=COLOR_Z)
        # Issue 30: Place equation at A4 for better alignment with the visual area
        self.place_at_grid(eq_z, "A4", scale_factor=0.9)
        self.play(Write(eq_z))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # This equation forms a diagonal line on the plane.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_LINE)
        
        # Plane area: B2 to F6. Visuals start at Col 2 to avoid lecture text (L002)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4.0,
            y_length=4.0,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "B2", "F6")
        
        # Labels for X and Y axis (Issue 31, 32)
        x_label = Text("X", font_size=20, color=WHITE)
        y_label = Text("Y", font_size=20, color=WHITE)
        # Position at axis tips as suggested by critic
        self.place_at_grid(x_label, "F6", scale_factor=0.8)
        x_label.shift(RIGHT * 0.4 + DOWN * 0.2)
        self.place_at_grid(y_label, "B2", scale_factor=0.8)
        y_label.shift(UP * 0.4 + LEFT * 0.2)
        
        # Diagonal line x + y = z. Initialize at z=2.0.
        z_init = 2.0
        diag_line = axes.plot(lambda x: z_init - x, x_range=[0, z_init], color=COLOR_LINE)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(diag_line))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # To find P(Z=z), we scan this specific line.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_PULSE)
        
        # Highlight current focus (L004: Indicate)
        self.play(Indicate(diag_line, color=COLOR_PULSE, scale_factor=1.1))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # We accumulate joint probabilities along this diagonal path.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_LINE)

        # Scanner Beam: Yellow line sliding across the plane
        z_tracker = ValueTracker(2.0)
        
        def update_line(obj):
            z = z_tracker.get_value()
            if z < 0.1: z = 0.1
            # Clip line domain to axes boundaries [0, 4]
            new_line = axes.plot(lambda x: z - x, x_range=[0, min(z, 4)], color=COLOR_LINE)
            obj.become(new_line)

        diag_line.add_updater(update_line)
        # [L024] Use rate_functions module prefix
        self.play(
            z_tracker.animate.set_value(0.5), 
            run_time=1.2, 
            rate_func=rate_functions.ease_in_out_quad
        )
        self.play(
            z_tracker.animate.set_value(3.5), 
            run_time=1.8, 
            rate_func=rate_functions.ease_in_out_quad
        )
        diag_line.remove_updater(update_line)
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # This scan explains how the sum's distribution emerges.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_BARS)

        # Show white bars representing joint probability accumulation along the scan line
        z_final = 3.5
        x_points = np.linspace(0.4, 3.1, 7)
        bars = VGroup()
        for xp in x_points:
            yp = z_final - xp
            # Mock joint density calculation (f(x,y))
            dist_to_center = np.sqrt((xp-1.75)**2 + (yp-1.75)**2)
            bar_height = np.exp(-dist_to_center**2) * 1.5
            
            p_start = axes.c2p(xp, yp)
            p_end = p_start + UP * bar_height
            bar_seg = Line(p_start, p_end, color=COLOR_BARS, stroke_width=4)
            bars.add(bar_seg)
            
        self.play(Create(bars, lag_ratio=0.15), run_time=2.0)
        self.wait(2.0)
