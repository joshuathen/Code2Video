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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "Summary: The Calculus Loop"
        lecture_lines = [
            "- Derivatives and integrals form a beautiful, continuous cycle.",
            "- Differentiation breaks a whole down into its parts.",
            "- Integration builds those parts back into a whole.",
            "- They are two sides of the same mathematical coin.",
            "- Master this connection to unlock the power of calculus."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # "Derivatives and integrals form a beautiful, continuous cycle."
        self.lecture[0].set_color(WHITE)
        
        # Infinity symbol (Lemniscate of Bernoulli)
        # x = a * cos(t) / (1 + sin^2(t)), y = a * sin(t) * cos(t) / (1 + sin^2(t))
        a_val = 1.8
        infinity_func = lambda t: np.array([
            a_val * np.cos(t) / (1 + (np.sin(t))**2),
            a_val * np.sin(t) * np.cos(t) / (1 + (np.sin(t))**2),
            0
        ])
        infinity_track = ParametricFunction(infinity_func, t_range=[0, TAU], color=WHITE)
        # Resolve Issue 49: Scale factor changed from 0.9 to 0.8
        self.place_in_area(infinity_track, 'B1', 'E6', scale_factor=0.8)
        
        self.play(Create(infinity_track), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Differentiation breaks a whole down into its parts."
        self.lecture[1].set_color("#00FF00")
        
        derivatives_label = Text("Derivatives\n(Rates)", font_size=20, color="#00FF00")
        # Resolve Issue 47: Grid position changed to 'B2' and scale factor to 0.7
        self.place_at_grid(derivatives_label, 'B2', scale_factor=0.7)
        
        self.play(Write(derivatives_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Integration builds those parts back into a whole."
        self.lecture[2].set_color("#FFFF00")
        
        integrals_label = Text("Integrals\n(Totals)", font_size=20, color="#FFFF00")
        # Resolve Issue 48: Grid position changed to 'B5' and scale factor to 0.7
        self.place_at_grid(integrals_label, 'B5', scale_factor=0.7)
        
        self.play(Write(integrals_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "They are two sides of the same mathematical coin."
        self.lecture[3].set_color("#FFD700")
        
        # Glowing spark travels around the track from left to right.
        # Lemniscate param: t=0 is rightmost tip, t=PI is leftmost tip.
        # t=PI/2 is the center (0,0) coming from bottom loop to top loop.
        # Proportion 0.5 on the track corresponds to t=PI (left loop tip).
        # Proportion 0.0/1.0 corresponds to t=0/TAU (right loop tip).
        spark_core = Dot(color="#FFD700", radius=0.08)
        spark_glow = Dot(color="#FFD700", radius=0.18, fill_opacity=0.3)
        spark = VGroup(spark_glow, spark_core)
        
        # Start at left loop (proportion 0.5)
        spark.move_to(infinity_track.point_from_proportion(0.5))
        self.add(spark)
        
        # Use ValueTracker for smooth continuous movement
        prop_tracker = ValueTracker(0.5)
        spark.add_updater(lambda m: m.move_to(infinity_track.point_from_proportion(prop_tracker.get_value())))
        
        # Move from left loop (0.5) to right loop (1.0)
        self.play(prop_tracker.animate.set_value(1.0), run_time=2.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # "Master this connection to unlock the power of calculus."
        self.lecture[4].set_color("#FFD700")
        
        # Complete the circuit: travels from right loop (0.0) back to left loop (0.5).
        # We jump the tracker to 0.0 because point_from_proportion(1.0) == point_from_proportion(0.0).
        prop_tracker.set_value(0.0)
        self.play(prop_tracker.animate.set_value(0.5), run_time=2.5, rate_func=linear)
        
        spark.clear_updaters()
        self.wait(3)
