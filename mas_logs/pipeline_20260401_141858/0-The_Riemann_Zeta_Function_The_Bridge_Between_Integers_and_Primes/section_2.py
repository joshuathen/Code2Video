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

class Section2Scene(TeachingScene):
    """
    Manim CE v0.19.0 script formally introducing the Riemann Zeta Function.
    Features formula display, an interactive slider for 's', and a dynamic graph.
    """
    def construct(self):
        # Setup Title and Lecture Lines
        lecture_lines = [
            "The Zeta Function sums reciprocals raised to power s.",
            "The value of s controls how fast terms shrink.",
            "Larger s values force the series to converge faster.",
            "When s equals two, we see the Basel landscape.",
            "This function maps s to an infinite summation."
        ]
        self.setup_layout("Defining the Zeta Function ζ(s)", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display the formula ζ(s) = ∑ 1/n^s in the center of the right panel
        formula = Text("ζ(s) = ∑ (1/nˢ)", font_size=42, color=WHITE)
        # Using A1 to F6 to center it initially in the right panel (Issue 37 logic)
        self.place_in_area(formula, "A1", "F6", scale_factor=1.0)
        
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Move formula to top and introduce slider for s ranging from 1.1 to 5.
        
        # Relocate formula to top area (Issue 35 logic)
        target_formula = formula.copy()
        self.place_in_area(target_formula, "A1", "B6", scale_factor=0.8)
        
        # Slider setup
        s_tracker = ValueTracker(2.0)
        slider_line = Line(LEFT, RIGHT, color=GRAY).scale(2.5)
        self.place_in_area(slider_line, "C1", "C6")
        
        slider_pointer = Triangle(color=YELLOW).scale(0.15).rotate(PI)
        s_label = Text("s = ", font_size=24, color=YELLOW)
        s_val_num = DecimalNumber(s_tracker.get_value(), num_decimal_places=2, font_size=24, color=YELLOW, mob_class=Text)
        s_indicator = VGroup(s_label, s_val_num).arrange(RIGHT, buff=0.1)
        
        def update_pointer(mob):
            # Map tracker value [1.1, 5.0] to slider line proportion [0, 1]
            alpha = (s_tracker.get_value() - 1.1) / (5.0 - 1.1)
            mob.move_to(slider_line.point_from_proportion(alpha) + UP * 0.2)
            
        slider_pointer.add_updater(update_pointer)
        s_val_num.add_updater(lambda d: d.set_value(s_tracker.get_value()))
        s_indicator.add_updater(lambda m: m.next_to(slider_pointer, UP, buff=0.1))

        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(
            Transform(formula, target_formula),
            Create(slider_line),
            FadeIn(slider_pointer),
            FadeIn(s_indicator)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a graph of f(n) = 1/n^s and animate the points as the slider s is dragged.
        
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.2, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "font_size": 18, "color": GRAY}
        )
        # Position graph in lower right area (Issue 36 logic)
        self.place_in_area(axes, "D1", "F6", scale_factor=0.9)
        
        def get_points_group(s_val):
            pts = VGroup()
            for n in range(1, 11):
                y = 1 / (n ** s_val)
                pts.add(Dot(axes.c2p(n, y), radius=0.06, color=GREEN))
            return pts

        dynamic_points = always_redraw(lambda: get_points_group(s_tracker.get_value()))
        
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(Create(axes), Create(dynamic_points))
        self.wait(0.5)
        
        # Increase s to show faster convergence
        self.play(s_tracker.animate.set_value(5.0), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight points at s=2 (Basel Problem)
        
        self.play(self.lecture[3].animate.set_color("#00AEFF"))
        self.play(s_tracker.animate.set_value(2.0), run_time=1.5)
        
        basel_text = Text("ζ(2) ≈ 1.645", font_size=24, color="#00AEFF")
        self.place_at_grid(basel_text, "D5")
        
        # Subtle highlight effect
        basel_glow = VGroup(*[
            Dot(axes.c2p(n, 1/(n**2)), radius=0.12, color="#00AEFF", fill_opacity=0.3) 
            for n in range(1, 11)
        ])
        
        self.play(FadeIn(basel_text), FadeIn(basel_glow))
        self.wait(2)
        self.play(FadeOut(basel_glow))

        # === Animation for Lecture Line 5 ===
        # Slide s towards 1 to show divergence (Harmonic series limit)
        
        self.play(self.lecture[4].animate.set_color(RED))
        
        divergence_text = Text("Diverges as s → 1", font_size=24, color=RED)
        self.place_at_grid(divergence_text, "E5")
        
        self.play(FadeOut(basel_text))
        self.play(
            s_tracker.animate.set_value(1.1),
            FadeIn(divergence_text),
            run_time=3
        )
        self.wait(2)

        # Final cleanup and stabilization (Issue 37 logic: ensure no overlap with left side)
        self.play(
            FadeOut(axes), 
            FadeOut(dynamic_points), 
            FadeOut(slider_line), 
            FadeOut(slider_pointer), 
            FadeOut(s_indicator),
            FadeOut(divergence_text)
        )
        
        final_formula_display = formula.copy()
        # Ensure final state is centered in right panel
        self.place_in_area(final_formula_display, "A1", "F6", scale_factor=1.2)
        self.play(Transform(formula, final_formula_display))
        self.wait(2)
