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
        # Fetching data from storyboard
        title_text = "Defining the PDE: The 'Curly d' (∂) Notation"
        lecture_lines = [
            "We use the curly 'del' symbol.",
            "Equations relate rates of change together.",
            "We solve for the unknown function 'u'."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Initialize lecture line colors
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Large partial derivative symbol partial appears in center in #FFFFFF.
        self.lecture[0].set_color(WHITE)
        partial_symbol = MathTex(r"\partial", color="#FFFFFF", font_size=160)
        # Resolved Issue 25: Move partial_symbol to C5-D6 to avoid overlap with later formula
        self.place_in_area(partial_symbol, "C5", "D6", scale_factor=1.0)
        
        self.play(Write(partial_symbol))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Display the relationship 'Time Rate = Curvature' in #FFD700.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FFD700")
        
        relation_text = Text("Time Rate = Curvature", color="#FFD700", font_size=28)
        # Resolved Issue 26: Adjust relation_text area to A2-A4 to reduce visual clutter
        self.place_in_area(relation_text, "A2", "A4", scale_factor=1.0)
        
        formula = MathTex(
            r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}", 
            color="#FFD700", 
            font_size=42
        )
        # Resolved Issue 26: Adjust formula area to B2-B4 to reduce visual clutter
        self.place_in_area(formula, "B2", "B4", scale_factor=1.0)

        self.play(
            FadeIn(relation_text),
            FadeIn(formula),
            partial_symbol.animate.scale(0.3).move_to(self.grid["A1"]),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Show a 3D surface plot changing shape based on its curvature.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE)
        
        # Simulating a surface using multiple curves for a 3D effect in 2D
        time_tracker = ValueTracker(0)
        
        def get_surface_curves(t_val):
            curves = VGroup()
            num_curves = 8
            for i in range(num_curves):
                z = i / (num_curves - 1) * 2 - 1 # from -1 to 1 depth
                # Projection perspective offset
                offset_x = z * 0.4
                offset_y = z * 0.2
                
                # Use Heat Equation solution form: amplitude = exp(-k*t)
                amplitude = 1.2 * np.exp(-t_val)
                
                curve = ParametricFunction(
                    lambda x: np.array([
                        x + offset_x, 
                        amplitude * np.sin(x) + offset_y, 
                        0
                    ]),
                    t_range=[-PI, PI],
                    color=BLUE_C,
                    stroke_opacity=1.0 - abs(z)*0.3
                )
                curves.add(curve)
            return curves

        # Build initial surface
        surface_plot = get_surface_curves(0)
        # Resolved Issue 27: Adjust surface_plot area to D2-F4 for better grid balance
        self.place_in_area(surface_plot, "D2", "F4", scale_factor=0.7)
        
        # Calculate area center for the updater based on new grid area
        tl_pos = self.grid["D2"]
        br_pos = self.grid["F4"]
        surface_center = np.array([(tl_pos[0] + br_pos[0]) / 2, (tl_pos[1] + br_pos[1]) / 2, 0])

        # Persistent updater for the "3D" surface
        def surface_updater(mob):
            t_curr = time_tracker.get_value()
            new_mob = get_surface_curves(t_curr)
            new_mob.scale(0.7).move_to(surface_center)
            mob.become(new_mob)

        surface_plot.add_updater(surface_updater)
        
        self.play(Create(surface_plot))
        # Animate the "flattening" representing heat dissipation / curvature-driven change
        self.play(time_tracker.animate.set_value(2.5), run_time=5, rate_func=linear)
        self.wait(2)
