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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Tautochrone Property (Bonus Insight)", 
                         ["This curve has another amazing secret property.", 
                          "Objects released from any point finish together.", 
                          "This is the \"equal time\" or tautochrone property."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Cycloid parameters: inverted cycloid
        r = 0.6
        # Parametric form: x = r(t + sin(t)), y = r(1 - cos(t))
        # t in [-PI, PI]
        cycloid_func = lambda t: np.array([r * (t + np.sin(t)), r * (1 - np.cos(t)), 0])
        
        # Create curve
        pink_curve = ParametricFunction(cycloid_func, t_range=[-PI, PI], color="#FF69B4")
        # Center initially to avoid offset issues before placing
        pink_curve.move_to(ORIGIN)
        
        # Place in area as per Issue 39: B1 to F6 with scale factor 1.2
        self.place_in_area(pink_curve, "B1", "F6", scale_factor=1.2)
        
        # Store the offset for the marbles to follow the same path relative to curve position
        # Since we scaled the curve by 1.2 in place_in_area, we need that scaling for points too.
        curve_center_offset = pink_curve.get_center()

        self.play(Create(pink_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)

        # Asset path from Issue 28
        marble_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marbles.svg"
        
        # Initial theta values for three marbles (spread across the left half)
        theta_starts = [-PI, -2*PI/3, -PI/3]
        marbles = VGroup()
        for theta in theta_starts:
            # Create instance of the marble asset
            marble = SVGMobject(marble_asset_path)
            marble.scale(0.15)
            # Position it at the starting point on the cycloid, applying the 1.2 scale factor
            pos = cycloid_func(theta) * 1.2 + curve_center_offset 
            marble.move_to(pos)
            marbles.add(marble)

        # Tint the marbles differently to tell them apart
        marbles[0].set_color(BLUE_B)
        marbles[1].set_color(GREEN_B)
        marbles[2].set_color(RED_B)

        self.play(FadeIn(marbles))
        self.wait(1)

        # Physics: Simple Harmonic Motion in arc length s
        # s(t) = s0 * cos(omega * t), where s = 4r * sin(theta/2)
        # Relationship: theta(t) = 2 * arcsin(sin(theta0/2) * cos(omega * t))
        
        omega = 1.0 
        time_tracker = ValueTracker(0)

        # Persistent updaters for smooth animation
        def get_update_fn(start_theta):
            def update_fn(m):
                t = time_tracker.get_value()
                # Marbles converge at t = PI / (2 * omega)
                t_constrained = min(t, PI / (2 * omega))
                
                # Desired theta at time t
                val = np.sin(start_theta / 2) * np.cos(omega * t_constrained)
                val = np.clip(val, -1, 1)
                theta_t = 2 * np.arcsin(val)
                
                # Update position based on parametric curve and scaling
                new_pos = cycloid_func(theta_t) * 1.2 + curve_center_offset
                m.move_to(new_pos)
            return update_fn

        for i, m in enumerate(marbles):
            m.add_updater(get_update_fn(theta_starts[i]))

        # Release the marbles!
        self.play(
            time_tracker.animate.set_value(PI / (2 * omega)), 
            run_time=3, 
            rate_func=linear
        )
        self.wait(0.5)

        # Clear updaters for stability
        for m in marbles:
            m.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GOLD)
        
        # Convergence point is at theta = 0
        flash_point = cycloid_func(0) * 1.2 + curve_center_offset
        self.play(Flash(flash_point, color=WHITE, line_length=0.3, num_lines=12))
        
        self.wait(3)
