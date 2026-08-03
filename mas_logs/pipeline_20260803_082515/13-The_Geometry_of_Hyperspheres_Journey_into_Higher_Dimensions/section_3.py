from manim import *
import math

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
    def n_sphere_vol(self, n):
        # Volume of n-sphere with radius R=1
        if n < 0: return 0.0
        if n == 0: return 1.0
        return (math.pi ** (n / 2)) / math.gamma(n / 2 + 1)

    def construct(self):
        title_text = "The Shrinking Volume Paradox"
        lecture_lines = [
            "Volume initially increases as we add dimensions.",
            "It peaks at the fifth dimension then drops.",
            "High-dimensional volume rapidly approaches zero.",
            "The hypercube contains almost all the available space.",
            "Hyperspheres become surprisingly empty in higher dimensions."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color definitions
        color_1 = BLUE_B
        color_peak = "#FFD700"  # Gold
        color_cube = WHITE
        color_empty = RED_B

        # === Animation for Lecture Line 1 ===
        # A plot with 'Dimension' on X and 'Volume' on Y appears.
        # The volume curve rises from d=1 to d=5, peaking at d=5.
        axes = Axes(
            x_range=[0, 21, 5],
            y_range=[0, 6, 2],
            x_length=4.2,
            y_length=2.8,
            axis_config={"include_tip": False, "font_size": 18},
        )
        labels = axes.get_axis_labels(x_label="n", y_label="Vol")
        graph_group = VGroup(axes, labels)
        # Fixed: Move from C2 to C3 to avoid clutter (Issue #24)
        self.place_in_area(graph_group, "C3", "F6", scale_factor=1.0)
        
        curve_rise = axes.plot(
            lambda x: self.n_sphere_vol(x),
            x_range=[0, 5.25],
            color=color_1
        )

        self.play(self.lecture[0].animate.set_color(color_1))
        self.play(Create(axes), FadeIn(labels), run_time=1)
        self.play(Create(curve_rise), run_time=1.5)
        self.wait(0.2)

        # === Animation for Lecture Line 2 ===
        # Highlight the peak at d=5 with a gold star icon.
        # The maximum volume for a unit n-sphere occurs at n = 5.256...
        peak_x = 5.25
        peak_val = self.n_sphere_vol(peak_x)
        star = Star(n=5, outer_radius=0.12, inner_radius=0.06, color=color_peak, fill_opacity=1)
        star.move_to(axes.c2p(peak_x, peak_val))

        self.play(self.lecture[1].animate.set_color(color_peak))
        self.play(FadeIn(star, scale=0.5), run_time=0.8)
        self.wait(0.2)

        # === Animation for Lecture Line 3 ===
        # The curve falls sharply toward zero from d=6 to d=20.
        curve_fall = axes.plot(
            lambda x: self.n_sphere_vol(x),
            x_range=[5.25, 20],
            color=color_1
        )
        
        self.play(self.lecture[2].animate.set_color(color_1))
        self.play(Create(curve_fall), run_time=2)
        self.wait(0.2)

        # === Animation for Lecture Line 4 ===
        # Show a unit sphere and a cube metaphor.
        # Square of side 2 (contains circle of radius 1)
        square = Square(side_length=1.2, color=color_cube)
        circle = Circle(radius=0.6, color=color_1, fill_opacity=0.3)
        
        cube_label = Text("Hypercube", font_size=14, color=color_cube)
        sphere_label = Text("Hypersphere", font_size=14, color=color_1)
        
        metaphor_group = VGroup(square, circle)
        # Fixed: Move from A2 to A3 for better alignment (Issue #25)
        self.place_in_area(metaphor_group, "A3", "B4", scale_factor=1.0)
        
        cube_label.next_to(square, UP, buff=0.1)
        sphere_label.next_to(square, DOWN, buff=0.1)

        self.play(self.lecture[3].animate.set_color(color_cube))
        self.play(
            Create(square), 
            Create(circle), 
            FadeIn(cube_label), 
            FadeIn(sphere_label),
            run_time=1.2
        )
        self.wait(0.2)

        # === Animation for Lecture Line 5 ===
        # Hyperspheres become surprisingly empty in higher dimensions.
        # The sphere shrinks to a tiny dot as the 'Volume' label drops to 0.
        dim_tracker = ValueTracker(2)
        
        vol_label_txt = Text("Sphere Vol:", font_size=16, color=color_empty)
        vol_val = DecimalNumber(self.n_sphere_vol(2), num_decimal_places=3, font_size=16, color=color_empty)
        vol_display = VGroup(vol_label_txt, vol_val).arrange(RIGHT, buff=0.1)
        # Fixed: Move from B6 to B5 to be closer to metaphor_group (Issue #26)
        self.place_at_grid(vol_display, "B5")

        original_radius = circle.radius
        # Ratio of volumes: V_n(1) / V_cube(n) = V_n(1) / 2^n. 
        # Reference ratio at n=2 is pi/4.
        r_ref = self.n_sphere_vol(2) / (2**2)
        
        def update_circle_size(m):
            n = dim_tracker.get_value()
            r_curr = self.n_sphere_vol(n) / (2**n)
            # Area ratio in 2D visualization = r_curr / r_ref
            # Radius scale factor = sqrt(area_ratio)
            sf = math.sqrt(r_curr / r_ref)
            m.set_width(2 * original_radius * max(sf, 0.01))
            
        def update_vol_val(m):
            m.set_value(self.n_sphere_vol(dim_tracker.get_value()))

        self.play(self.lecture[4].animate.set_color(color_empty))
        self.play(FadeIn(vol_display), run_time=0.8)
        
        circle.add_updater(update_circle_size)
        vol_val.add_updater(update_vol_val)
        
        self.play(dim_tracker.animate.set_value(20), run_time=4, rate_func=linear)
        self.wait(1.5)
        
        circle.remove_updater(update_circle_size)
        vol_val.remove_updater(update_vol_val)
