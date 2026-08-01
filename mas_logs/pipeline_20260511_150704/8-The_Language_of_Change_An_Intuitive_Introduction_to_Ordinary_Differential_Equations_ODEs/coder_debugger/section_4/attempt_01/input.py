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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initializing the layout with script lines
        lecture_lines = [
            "We can visualize solutions using a slope field.",
            "Every point shows a tiny arrow of direction.",
            "These arrows map the required slope everywhere.",
            "A solution is a path following these arrows.",
            "Like a boat drifting along a river's current."
        ]
        self.setup_layout("Visualizing Solutions: Slope Fields", lecture_lines)

        # Coordinate system for the slope field
        # Positioned in the B1-F6 grid area to avoid crowding the title (Issue 42)
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5.0,
            y_length=5.0,
            axis_config={"include_tip": True, "color": GREY_B, "stroke_width": 2},
            tips=False
        )
        self.place_in_area(axes, 'B1', 'F6', scale_factor=0.85)

        # ODE: dy/dx = y / 2
        def get_slope_line(x_val, y_val, length=0.22, color="#555555"):
            slope = y_val / 2
            angle = np.arctan(slope)
            p1 = axes.c2p(x_val - (length/2) * np.cos(angle), y_val - (length/2) * np.sin(angle))
            p2 = axes.c2p(x_val + (length/2) * np.cos(angle), y_val + (length/2) * np.sin(angle))
            return Line(p1, p2, color=color, stroke_width=2)

        slope_field = VGroup()
        for x in np.arange(-2.0, 2.1, 0.5):
            for y in np.arange(-2.0, 2.1, 0.5):
                slope_field.add(get_slope_line(x, y))

        # === Animation for Lecture Line 1 ===
        # Color: #555555 (Grey) for the field
        self.play(self.lecture[0].animate.set_color("#555555"))
        self.play(Create(axes), FadeIn(slope_field, lag_ratio=0.02), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight individual arrows
        self.play(self.lecture[1].animate.set_color("#555555"))
        # Selecting a few arrows in the middle
        highlight_indices = [40, 41, 42, 43, 44]
        highlight_arrows = VGroup(*[slope_field[i] for i in highlight_indices if i < len(slope_field)])
        self.play(highlight_arrows.animate.set_color(YELLOW).scale(1.4), run_time=0.8)
        self.play(highlight_arrows.animate.set_color("#555555").scale(1/1.4), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#555555"))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Tracing a single solution curve: y = e^(x/2)
        # Color: #0000FF (Blue)
        # Creating relative to axes ensures grid anchoring (Issue 43)
        self.play(self.lecture[3].animate.set_color("#0000FF"))
        
        curve_func = lambda x: np.exp(x/2)
        sol_curve = axes.plot(curve_func, x_range=[-2.5, 1.8], color="#0000FF", stroke_width=4)
        
        # Boat Asset (Issue 37)
        boat = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/boat.svg")
        boat.scale(0.12).set_color(WHITE)
        # Initial orientation
        initial_angle = np.arctan(0.5 * np.exp(-2.5/2))
        boat.rotate(initial_angle)
        boat.move_to(sol_curve.get_start())
        
        self.add(boat)
        self.play(
            Create(sol_curve), 
            MoveAlongPath(boat, sol_curve), 
            run_time=3.5, 
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Family of curves: y = C * e^(x/2)
        # Color: #FFFFFF (White)
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        dashed_curves = VGroup()
        c_values = [-1.5, -0.6, 0.4, 1.6]
        for c in c_values:
            # Adjusting x_range so curves stay within visual bounds (Issue 44)
            x_end = 2.2 if abs(c) < 1 else (1.2 if c > 0 else 2.2)
            curve = axes.plot(
                lambda x, c_val=c: c_val * np.exp(x/2), 
                x_range=[-2.5, x_end], 
                color=WHITE, 
                stroke_opacity=0.6
            )
            dashed_curves.add(curve)
            
        self.play(FadeIn(dashed_curves, lag_ratio=0.2), run_time=2.5)
        self.wait(2)
