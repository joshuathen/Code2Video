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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Differentiation and integration are inverse operations.",
            "They are like addition and subtraction for functions.",
            "Today, we bridge these two mathematical worlds."
        ]
        self.setup_layout("Introduction: The Inverse Twins", lecture_lines)
        
        # Colors
        COLOR_MACHINE = "#FFFFFF"
        COLOR_DERIVATIVE = "#FFD700"
        COLOR_INTEGRATION = "#1E90FF"
        COLOR_SHAPE = "#00FF00"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR), run_time=0.5)

        # Create Machine using Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg]
        machine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        machine_icon.set_color(COLOR_MACHINE)
        # Fix Issue 41: use A2 to E5 for better centering/spacing of the icon and label
        self.place_in_area(machine_icon, "A2", "E5", scale_factor=2.0)
        
        machine_label = Text("FUNCTION MACHINE", font_size=18, color=COLOR_MACHINE)
        # Position label relative to the machine icon
        machine_label.next_to(machine_icon, UP, buff=0.2)
        
        self.play(FadeIn(machine_icon), Write(machine_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )

        # Derivative Lever
        lever_deriv_base = Line(DOWN*0.5, UP*0.5, color=COLOR_DERIVATIVE)
        lever_deriv_knob = Dot(color=COLOR_DERIVATIVE).move_to(lever_deriv_base.get_top())
        lever_deriv = VGroup(lever_deriv_base, lever_deriv_knob)
        # Fix Issue 43: place at B2 to avoid being too close to the edge/center
        self.place_at_grid(lever_deriv, "B2")
        lever_deriv_text = Text("d/dx", font_size=20, color=COLOR_DERIVATIVE)
        lever_deriv_text.next_to(lever_deriv, UP, buff=0.1)

        # The Shape
        original_shape = Rectangle(width=2, height=2, fill_opacity=0.3, fill_color=COLOR_SHAPE, color=COLOR_SHAPE)
        self.place_in_area(original_shape, "C3", "D4")

        # The Slices
        num_slices = 8
        slices = VGroup(*[
            Rectangle(width=2/num_slices, height=2, fill_opacity=1.0, fill_color=COLOR_SHAPE, color=COLOR_SHAPE, stroke_width=1)
            for _ in range(num_slices)
        ]).arrange(RIGHT, buff=0.05)
        self.place_in_area(slices, "C3", "D4")

        self.play(FadeIn(lever_deriv), Write(lever_deriv_text), FadeIn(original_shape))
        self.wait(0.5)

        # Pull lever and slice
        self.play(lever_deriv_knob.animate.move_to(lever_deriv_base.get_bottom()), run_time=0.5)
        self.play(
            ReplacementTransform(original_shape, slices),
            run_time=1.5
        )
        self.play(lever_deriv_knob.animate.move_to(lever_deriv_base.get_top()), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )

        # Integration Lever
        lever_int_base = Line(DOWN*0.5, UP*0.5, color=COLOR_INTEGRATION)
        lever_int_knob = Dot(color=COLOR_INTEGRATION).move_to(lever_int_base.get_top())
        lever_int = VGroup(lever_int_base, lever_int_knob)
        # Fix Issue 42: place at B5 to avoid edge cutoff
        self.place_at_grid(lever_int, "B5")
        lever_int_text = Text("∫ dx", font_size=20, color=COLOR_INTEGRATION)
        lever_int_text.next_to(lever_int, UP, buff=0.1)

        self.play(FadeIn(lever_int), Write(lever_int_text))
        self.wait(0.5)

        # Pull lever and glue
        self.play(lever_int_knob.animate.move_to(lever_int_base.get_bottom()), run_time=0.5)
        
        # Reform original shape
        reformed_shape = Rectangle(width=2, height=2, fill_opacity=0.3, fill_color=COLOR_SHAPE, color=COLOR_SHAPE)
        self.place_in_area(reformed_shape, "C3", "D4")

        self.play(
            ReplacementTransform(slices, reformed_shape),
            run_time=1.5
        )
        self.play(lever_int_knob.animate.move_to(lever_int_base.get_top()), run_time=0.5)
        
        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)
        self.wait(2)
