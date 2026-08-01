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
        # Define the lecture lines and title from Shared State
        title = "Beyond the Slope: The Mapping Paradigm"
        lines = [
            "Beyond slopes, let’s view functions as transformations of space.",
            "Input points on one line map to an output line.",
            "Watch the shadow move as the input point slides."
        ]
        self.setup_layout(title, lines)
        
        # Mapping function: f(x) = 1.5x (Local scaling factor demo)
        def f_func(x):
            return 1.5 * x

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color("#FFFF00") # Light yellow
        
        # Create Input and Output lines
        # Input line (X): Top line (#87CEEB)
        # Issue 37: Area B3-B6
        input_line = NumberLine(
            x_range=[0, 5, 1],
            length=3.5,
            color="#87CEEB",
            include_numbers=True,
            label_direction=UP,
            font_size=16
        )
        # Output line (Y): Bottom line (#FF6347)
        # Issue 37: Area E3-E6
        output_line = NumberLine(
            x_range=[0, 8, 2],
            length=3.5,
            color="#FF6347",
            include_numbers=True,
            label_direction=DOWN,
            font_size=16
        )
        
        self.place_in_area(input_line, "B3", "B6")
        self.place_in_area(output_line, "E3", "E6")
        
        # Labels - Issue 36: Position at B2, E2. Issue 38: Scale 0.6
        input_label = Text("Input (X)", font_size=20, color="#87CEEB")
        output_label = Text("Output (Y)", font_size=20, color="#FF6347")
        
        self.place_at_grid(input_label, "B2", scale_factor=0.6)
        self.place_at_grid(output_label, "E2", scale_factor=0.6)
        
        self.play(
            Create(input_line),
            Create(output_line),
            Write(input_label),
            Write(output_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.lecture[1].set_color("#87CEEB")
        
        x_start = 2.0
        x_tracker = ValueTracker(x_start)
        
        # Input point on top line
        dot_input = Dot(color="#87CEEB")
        dot_input.move_to(input_line.number_to_point(x_tracker.get_value()))
        
        label_x = MathTex("x=2", font_size=24, color="#87CEEB")
        label_x.next_to(dot_input, UP, buff=0.2)
        
        # Output shadow asset - Issue 19
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/shadow.svg]
        shadow_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/shadow.svg"
        shadow = SVGMobject(shadow_path)
        shadow.set_color("#FF6347")
        shadow.scale(0.3)
        shadow.move_to(output_line.number_to_point(f_func(x_start)))
        
        label_y = MathTex("f(2)", font_size=24, color="#FF6347")
        label_y.next_to(shadow, DOWN, buff=0.2)

        # Curved arrow connecting input to output
        mapping_arrow = CurvedArrow(
            dot_input.get_center() + DOWN * 0.1,
            shadow.get_center() + UP * 0.1,
            angle=-TAU / 8,
            color="#FFFFFF"
        ).set_stroke(width=2, opacity=0.8)
        
        self.play(FadeIn(dot_input), Write(label_x))
        self.wait(0.5)
        self.play(Create(mapping_arrow))
        self.wait(0.5)
        self.play(FadeIn(shadow), Write(label_y))
        self.wait(1.5)
        
        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.lecture[2].set_color("#FF6347")
        
        # Prepare for movement by switching to dynamic labels
        val_x = DecimalNumber(x_start, num_decimal_places=1, font_size=20, color="#87CEEB")
        label_x_group = VGroup(Text("x=", font_size=20, color="#87CEEB"), val_x).arrange(RIGHT, buff=0.1)
        label_x_group.next_to(dot_input, UP, buff=0.2)
        
        label_y_text = Text("f(x)", font_size=20, color="#FF6347")
        label_y_text.next_to(shadow, DOWN, buff=0.2)
        
        self.play(
            FadeOut(label_x), FadeIn(label_x_group),
            FadeOut(label_y), FadeIn(label_y_text),
            run_time=0.5
        )

        # Set up persistent mobject updaters
        dot_input.add_updater(lambda d: d.move_to(input_line.number_to_point(x_tracker.get_value())))
        label_x_group.add_updater(lambda l: l.next_to(dot_input, UP, buff=0.2))
        val_x.add_updater(lambda v: v.set_value(x_tracker.get_value()))
        
        shadow.add_updater(lambda s: s.move_to(output_line.number_to_point(f_func(x_tracker.get_value()))))
        label_y_text.add_updater(lambda l: l.next_to(shadow, DOWN, buff=0.2))
        
        # Update mapping arrow shape
        mapping_arrow.add_updater(lambda m: m.become(
            CurvedArrow(
                dot_input.get_center() + DOWN * 0.1,
                shadow.get_center() + UP * 0.1,
                angle=-TAU / 8,
                color="#FFFFFF"
            ).set_stroke(width=2, opacity=0.8)
        ))
        
        # Slide point from x=2 to x=3 to demonstrate transformation
        self.play(x_tracker.animate.set_value(3.0), run_time=3)
        self.wait(2.0)
