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
    def construct(self):
        # Initial Setup
        title = "Beyond the Slope: The Linear Zoom"
        lines = [
            "Zooming in, any curve looks like a straight line.",
            "Visualize this local behavior on parallel lines.",
            "A small change in the input is called dx.",
            "The output change df is twice as large here.",
            "The derivative is this local scaling factor."
        ]
        self.setup_layout(title, lines)

        # Colors for elements
        color_input = "#FFFF00"  # Yellow
        color_output = "#FF00FF" # Magenta
        color_scaling = "#00FF00" # Green
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-1, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLUE_C, "include_tip": True}
        )
        parabola = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 2.5], color=WHITE)
        point_dot = Dot(axes.c2p(1, 0.5), color=RED)
        graph_group = VGroup(axes, parabola, point_dot)
        
        # Resolve Issue 26: Avoid occlusion
        self.place_in_area(graph_group, 'C3', 'F6', scale_factor=0.8)
        
        # Resolve Issue 21: Asset integration
        camera_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/camera.svg", fill_color=WHITE)
        self.place_at_grid(camera_icon, "C3", scale_factor=0.3)
        
        self.play(Create(axes), Create(parabola))
        self.play(FadeIn(point_dot), FadeIn(camera_icon))
        
        zoom_point = axes.c2p(1, 0.5)
        self.play(
            graph_group.animate.scale(8, about_point=zoom_point),
            camera_icon.animate.scale(0.5).move_to(self.grid["B3"]), # Keep camera visible but smaller
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(FadeOut(graph_group), FadeOut(camera_icon))
        
        # Resolve Issue 27: Position mapping lines to avoid encroachment
        input_line = NumberLine(x_range=[0, 2, 0.5], length=4.5, include_numbers=True, label_constructor=Text, color=WHITE)
        output_line = NumberLine(x_range=[0, 2, 0.5], length=4.5, include_numbers=True, label_constructor=Text, color=WHITE)
        
        self.place_in_area(input_line, 'B2', 'B6')
        self.place_in_area(output_line, 'E2', 'E6')
        
        input_label = Text("Input", font_size=18).next_to(input_line, UP, buff=0.1)
        output_label = Text("Output", font_size=18).next_to(output_line, DOWN, buff=0.1)
        
        self.play(FadeIn(input_line), FadeIn(output_line), FadeIn(input_label), FadeIn(output_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_input))
        
        dx_segment = Line(
            input_line.n2p(0.75), input_line.n2p(1.25), 
            color=color_input, stroke_width=8
        )
        # Resolve Issue 27: Position dx text
        dx_text = Text("dx", color=color_input, font_size=24)
        self.place_at_grid(dx_text, 'A4')
        
        self.play(Create(dx_segment), Write(dx_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color_output))
        
        df_segment = Line(
            output_line.n2p(0.5), output_line.n2p(1.5), 
            color=color_output, stroke_width=8
        )
        # Resolve Issue 27: Position df text
        df_text = Text("df", color=color_output, font_size=24)
        self.place_at_grid(df_text, 'F4')
        
        self.play(Create(df_segment), Write(df_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color_scaling))
        
        arrow_start = Arrow(dx_segment.get_start(), df_segment.get_start(), color=GRAY, buff=0.1, stroke_width=2)
        arrow_end = Arrow(dx_segment.get_end(), df_segment.get_end(), color=GRAY, buff=0.1, stroke_width=2)
        
        # Resolve Issue 28: Expand scaling factor area
        scaling_text = Text("Scaling Factor = 2", font_size=24, color=color_scaling)
        self.place_in_area(scaling_text, 'C2', 'D5', scale_factor=0.8)
        
        self.play(GrowArrow(arrow_start), GrowArrow(arrow_end))
        self.play(Write(scaling_text))
        self.wait(2)
