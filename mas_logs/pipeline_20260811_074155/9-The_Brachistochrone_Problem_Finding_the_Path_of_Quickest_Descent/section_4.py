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
        title = "The Core Logic: Bernoulli's Insight"
        lecture_lines = [
            "Bernoulli treated the sliding particle as a light beam.",
            "The refractive index changes at every level of depth.",
            "Use the formula sin theta over velocity equals constant.",
            "As velocity increases, the angle must also increase.",
            "This optical analogy reveals the shape of quickest descent."
        ]
        self.setup_layout(title, lecture_lines)

        # Helper for Cycloid
        R = 0.8
        def cycloid_func(t):
            start_x = self.grid["A1"][0]
            start_y = self.grid["A1"][1]
            return np.array([
                start_x + R * (t - np.sin(t)),
                start_y - R * (1 - np.cos(t)),
                0
            ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        curve = ParametricFunction(cycloid_func, t_range=[0, np.pi], color=BLUE)
        num_layers = 5
        t_vals = np.linspace(0, np.pi, num_layers + 1)
        light_path_pts = [cycloid_func(t) for t in t_vals]
        light_path = VGroup(*[
            Line(light_path_pts[i], light_path_pts[i+1], color=WHITE, stroke_width=2)
            for i in range(num_layers)
        ])
        
        self.play(Create(curve), run_time=2)
        self.play(Create(light_path), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        bands = VGroup()
        colors = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A]
        for i in range(5):
            band_y = self.grid["A1"][1] - i * 0.8
            rect = Rectangle(width=6, height=0.8, fill_opacity=0.2, fill_color=colors[i], stroke_width=0)
            rect.move_to(np.array([3.0, band_y - 0.4, 0]))
            bands.add(rect)
            
        self.play(FadeIn(bands), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        formula = MathTex(r"\frac{\sin(\theta)}{v} = \text{constant}", color="#FFFF00")
        # Fixed positioning per issue 31: Move to A5-B6 area
        self.place_in_area(formula, 'A5', 'B6', scale_factor=0.7)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        t_tracker = ValueTracker(0.5)
        
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda d: d.move_to(cycloid_func(t_tracker.get_value())))
        
        # Persistent mobjects for updater to avoid recreating Text/MathTex inside frames
        v_line = DashedLine(color=GREY)
        arc = Arc(radius=0.3, color=WHITE)
        theta_label = MathTex(r"\theta", font_size=24, color=WHITE)
        v_arrow = Arrow(color=ORANGE, buff=0)
        v_label = MathTex("v", font_size=24, color=ORANGE)
        
        def update_visuals(m):
            t = t_tracker.get_value()
            p = cycloid_func(t)
            
            dx = R * (1 - np.cos(t))
            dy = -R * np.sin(t)
            tangent_vec = np.array([dx, dy, 0])
            norm = np.linalg.norm(tangent_vec)
            if norm > 0:
                tangent_vec /= norm
            
            v_line.put_start_and_end_on(p + UP*0.5, p + DOWN*0.5)
            
            # Geometry updates are performant; redraw the arc specifically
            new_arc = Arc(radius=0.3, start_angle=-PI/2, angle=t/2, arc_center=p, color=WHITE)
            arc.become(new_arc)
            
            theta_label.next_to(arc, RIGHT, buff=0.1)
            v_arrow.put_start_and_end_on(p, p + tangent_vec * 0.8)
            v_label.next_to(v_arrow, RIGHT, buff=0.1)

        # Initial sync
        update_visuals(None)
        visuals = VGroup(v_line, arc, theta_label, v_arrow, v_label)
        visuals.add_updater(update_visuals)
        
        self.add(dot, visuals)
        
        self.play(t_tracker.animate.set_value(2.5), run_time=4, rate_func=linear)
        self.wait(1)

        # Scale formula for emphasis
        self.play(formula.animate.scale(1.2).set_color(GOLD), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(curve.animate.set_color("#FF00FF").set_stroke(width=6))
        self.wait(2)
