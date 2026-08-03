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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Inverse Concept: Working Backwards", 
            [
                "Integration is the inverse process of finding the derivative.",
                "If we have velocity, we can recover total distance.",
                "We call this process anti-differentiation or finding the integral."
            ]
        )
        
        # Colors for consistency
        LINE1_COLOR = WHITE
        LINE2_COLOR = "#87CEEB" # Light blue
        LINE3_COLOR = "#90EE90" # Light green
        MACHINE_COLOR = "#C0C0C0"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(LINE1_COLOR)
        
        # Create Function Machine
        machine_box = RoundedRectangle(corner_radius=0.2, width=3.5, height=4, color=MACHINE_COLOR, fill_opacity=0.1)
        # Fix Issue 35: Moved machine_box from 'B2'-'E5' to 'B3'-'E6' to avoid crowding lecture text.
        self.place_in_area(machine_box, "B3", "E6")
        
        machine_label = Text("FUNCTION MACHINE", font_size=20, color=MACHINE_COLOR)
        machine_label.move_to(machine_box.get_top() + DOWN * 0.4)
        
        # Levers
        derivative_lever = VGroup(
            Line(LEFT*0.3, RIGHT*0.3, color=LINE2_COLOR),
            Circle(radius=0.1, color=LINE2_COLOR, fill_opacity=1)
        ).move_to(machine_box.get_right() + LEFT*0.3 + UP*0.5)
        
        derivative_text = Text("Derivative", font_size=16, color=LINE2_COLOR)
        derivative_text.next_to(derivative_lever, LEFT, buff=0.1)
        
        integral_lever = VGroup(
            Line(LEFT*0.3, RIGHT*0.3, color=LINE3_COLOR),
            Circle(radius=0.1, color=LINE3_COLOR, fill_opacity=1)
        ).move_to(machine_box.get_right() + LEFT*0.3 + DOWN*0.5)
        
        integral_text = Text("Integral", font_size=16, color=LINE3_COLOR)
        integral_text.next_to(integral_lever, LEFT, buff=0.1)

        self.play(
            Create(machine_box),
            Write(machine_label),
            FadeIn(derivative_lever),
            FadeIn(derivative_text),
            FadeIn(integral_lever),
            FadeIn(integral_text),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(LINE2_COLOR)
        
        # x^2 enters
        input_x2 = MathTex("x^2", color=WHITE)
        # Fix Issue 36: Moved input_x2 start position from 'C1' to 'C2' to avoid overlap with lecture text.
        self.place_at_grid(input_x2, "C2")
        
        self.play(FadeIn(input_x2))
        self.play(input_x2.animate.move_to(machine_box.get_center()), run_time=1)
        
        # Lever pull animation
        self.play(
            derivative_lever.animate.shift(DOWN * 0.2),
            derivative_text.animate.set_color(YELLOW),
            rate_func=there_and_back,
            run_time=0.5
        )
        derivative_text.set_color(LINE2_COLOR)
        
        # 2x exits
        output_2x = MathTex("2x", color=LINE2_COLOR)
        output_2x.move_to(machine_box.get_center())
        
        self.play(
            ReplacementTransform(input_x2, output_2x),
            run_time=0.5
        )
        
        end_pos_2x = self.grid["C6"]
        self.play(output_2x.animate.move_to(end_pos_2x), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(LINE3_COLOR)
        
        # 2x enters back
        input_2x = MathTex("2x", color=LINE2_COLOR)
        # Fix Issue 37: Moved input_2x start position from 'D1' to 'D2' to avoid overlap with lecture text.
        self.place_at_grid(input_2x, "D2")
        
        self.play(FadeIn(input_2x))
        self.play(input_2x.animate.move_to(machine_box.get_center()), run_time=1)
        
        # Integral lever pull
        self.play(
            integral_lever.animate.shift(DOWN * 0.2),
            integral_text.animate.set_color(YELLOW),
            rate_func=there_and_back,
            run_time=0.5
        )
        integral_text.set_color(LINE3_COLOR)
        
        # x^2 exits
        output_x2_final = MathTex("x^2", color=LINE3_COLOR)
        output_x2_final.move_to(machine_box.get_center())
        
        self.play(
            ReplacementTransform(input_2x, output_x2_final),
            run_time=0.5
        )
        
        end_pos_x2 = self.grid["D6"]
        self.play(output_x2_final.animate.move_to(end_pos_x2), run_time=1)
        self.wait(2)
