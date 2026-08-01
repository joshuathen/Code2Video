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
        # Setup context
        title = "Summary and Real-World Application"
        lines = [
            "We've bridged the gap between calculus and dynamic systems.",
            "Matrix exponentials allow drones to remain stable in flight.",
            "Master this formula to solve complex multi-variable change."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line and show vertical flowchart
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        series_text = Text("Series", font_size=24, color=WHITE)
        diag_text = Text("Diagonalization", font_size=24, color=WHITE)
        flow_text = Text("System Flow", font_size=24, color=WHITE)
        
        self.place_at_grid(series_text, "B2", scale_factor=0.8)
        self.place_at_grid(diag_text, "D2", scale_factor=0.8)
        self.place_at_grid(flow_text, "F2", scale_factor=0.8)
        
        arrow1 = Arrow(series_text.get_bottom(), diag_text.get_top(), color=WHITE, buff=0.2)
        arrow2 = Arrow(diag_text.get_bottom(), flow_text.get_top(), color=WHITE, buff=0.2)
        
        self.play(Write(series_text))
        self.play(GrowArrow(arrow1), Write(diag_text))
        self.play(GrowArrow(arrow2), Write(flow_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition to Drone Stabilizer
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            FadeOut(series_text, diag_text, flow_text, arrow1, arrow2)
        )
        
        # Drone icon
        drone_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/drone.svg"
        drone = SVGMobject(drone_path).set_color("#00FF00")
        self.place_in_area(drone, "B3", "E5", scale_factor=1.2)
        
        # Thrust arrows with dynamic scaling via ValueTracker
        thrust_amp = ValueTracker(0)
        
        def create_motor_arrow(direction, anchor):
            arrow = Arrow(anchor, anchor + direction, color="#00FF00", buff=0)
            arrow.add_updater(lambda m: m.set_length(0.3 + 0.3 * np.sin(thrust_amp.get_value() + np.sum(anchor))))
            return arrow

        c = drone.get_center()
        offsets = [
            (UP + LEFT, c + LEFT * 0.8 + UP * 0.4),
            (UP + RIGHT, c + RIGHT * 0.8 + UP * 0.4),
            (DOWN + LEFT, c + LEFT * 0.8 + DOWN * 0.4),
            (DOWN + RIGHT, c + RIGHT * 0.8 + DOWN * 0.4)
        ]
        
        arrows = VGroup(*[create_motor_arrow(d, a) for d, a in offsets])
        
        self.play(DrawBorderThenFill(drone))
        self.play(Create(arrows))
        self.play(thrust_amp.animate.set_value(TAU * 2), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final Formula Display
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            FadeOut(drone, arrows)
        )
        
        # Fixed: Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        formula = Text("x(t) = e^At x(0)", color=WHITE, font_size=36)
        # Final center focus
        self.place_in_area(formula, "B1", "E6", scale_factor=1.8)
        
        self.play(Write(formula))
        self.play(formula.animate.scale(1.2))
        self.wait(2)
