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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Let's slide point Q closer to our target P.",
            "The gap between them, called h, shrinks toward zero.",
            "As Q approaches P, the secant line starts tilting.",
            "It transforms into a tangent line touching one point.",
            "This limit reveals the exact speed at point P."
        ]
        self.setup_layout("The Limit: Zooming into Infinity", lecture_lines)
        
        # Asset: Load icon
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        self.place_at_grid(based_icon, "F6", scale_factor=0.5)
        
        # Setup axes and curve
        # Area A2 to F6 covers the right side (Issue 29)
        axes = Axes(
            x_range=[0, 2.5, 1],
            y_range=[0, 6.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GREY_B}
        )
        self.place_in_area(axes, 'A2', 'F6', scale_factor=0.9)
        
        curve = axes.plot(lambda x: x**2, x_range=[0, 2.4], color=BLUE)
        
        # Point P is fixed at x=1
        x_p = 1
        y_p = x_p**2
        p_point = Dot(axes.c2p(x_p, y_p), color=WHITE)
        p_label = MathTex("P", color=WHITE, font_size=24)
        p_label.next_to(p_point, LEFT, buff=0.1)
        
        # h Tracker to move Point Q
        h_tracker = ValueTracker(1.2) # Initial distance from P
        
        # Point Q (#00FF00) slides down the curve toward point P (#FFFFFF)
        q_point = Dot(color="#00FF00")
        q_point.add_updater(lambda m: m.move_to(axes.c2p(x_p + h_tracker.get_value(), (x_p + h_tracker.get_value())**2)))
        
        q_label = MathTex("Q", color="#00FF00", font_size=24)
        q_label.add_updater(lambda m: m.next_to(q_point, UR, buff=0.1))

        # Yellow (#FFFF00) secant line rotates to track the moving point Q
        secant_line = Line(color=YELLOW)
        def update_secant(m):
            h = h_tracker.get_value()
            if h < 0.005:
                m.set_color("#00FF00")
                # Tangent line at x=1: y = 2x - 1.
                p1 = axes.c2p(0.4, 2*0.4 - 1)
                p2 = axes.c2p(2.2, 2*2.2 - 1)
                m.set_points_as_corners([p1, p2])
            else:
                m.set_color(YELLOW)
                # Secant through (1,1) and (1+h, (1+h)^2). Slope m_s = ( (1+h)^2 - 1 ) / h = 2+h
                m_s = 2 + h
                p1 = axes.c2p(0.4, m_s*(0.4 - x_p) + y_p)
                p2 = axes.c2p(2.2, m_s*(2.2 - x_p) + y_p)
                m.set_points_as_corners([p1, p2])
        
        secant_line.add_updater(update_secant)
        
        # White (#FFFFFF) text label shows the slope value decreasing towards 2.
        # Issue 30: place_at_grid(slope_label, 'A5', scale_factor=0.8)
        slope_label = Text("Slope:", font_size=24, color=WHITE)
        self.place_at_grid(slope_label, 'A5', scale_factor=0.8)
        
        slope_val = DecimalNumber(2 + 1.2, num_decimal_places=2, color=WHITE, font_size=24)
        slope_val.add_updater(lambda m: m.set_value(2 + h_tracker.get_value()))
        slope_val.next_to(slope_label, RIGHT, buff=0.2)
        
        # Cyan (#00FFFF) horizontal bracket shows the distance 'h' shrinking to zero.
        # Using always_redraw for the bracket as it's a structural component, not text.
        h_bracket = always_redraw(lambda: 
            BraceBetweenPoints(axes.c2p(x_p, 0), axes.c2p(x_p + h_tracker.get_value(), 0), color="#00FFFF")
            if h_tracker.get_value() > 0.05 else Line(axes.c2p(x_p, 0), axes.c2p(x_p, 0), stroke_width=0)
        )
        
        h_label = MathTex("h", color="#00FFFF", font_size=24)
        h_label.add_updater(lambda m: m.next_to(h_bracket, DOWN, buff=0.1))

        # === Animation for Lecture Line 1 ===
        # Let's slide point Q closer to our target P.
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Create(curve), FadeIn(p_point), FadeIn(p_label))
        self.play(FadeIn(q_point), FadeIn(q_label), Create(secant_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The gap between them, called h, shrinks toward zero.
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(h_bracket), FadeIn(h_label), FadeIn(slope_label), FadeIn(slope_val))
        self.play(h_tracker.animate.set_value(0.6), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # As Q approaches P, the secant line starts tilting.
        self.lecture[2].set_color(YELLOW)
        self.play(h_tracker.animate.set_value(0.2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # It transforms into a tangent line touching one point.
        self.lecture[3].set_color(YELLOW)
        self.play(h_tracker.animate.set_value(0.01), run_time=2)
        # At this point the updater makes it look like a tangent line (green)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This limit reveals the exact speed at point P.
        self.lecture[4].set_color(YELLOW)
        self.play(h_tracker.animate.set_value(0), run_time=1)
        self.play(FadeIn(based_icon)) # Asset integration
        self.wait(2)
