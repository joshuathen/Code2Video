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
        lecture_lines = [
            "Symmetry simplifies our mathematical \"stew\" significantly.",
            "Even functions mirror perfectly across the vertical axis.",
            "Odd functions exhibit rotational symmetry through the origin.",
            "Cosines are the \"Even Kings\" of the wave world.",
            "Sines serve as the \"Odd Queens\" in our series."
        ]
        self.setup_layout("The Symmetry Filter: Even and Odd Functions", lecture_lines)
        
        # Colors
        COLOR_EVEN = "#ADD8E6"
        COLOR_ODD = "#90EE90"

        # === Animation for Lecture Line 1 ===
        # Symmetry simplifies our mathematical "stew" significantly.
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Even functions mirror perfectly across the vertical axis.
        self.play(self.lecture[1].animate.set_color(COLOR_EVEN))
        
        axes_even = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 4, 1],
            x_length=2.0,
            y_length=1.5,
            axis_config={"include_tip": False, "color": GREY, "stroke_width": 2}
        )
        self.place_in_area(axes_even, "B1", "C3")
        
        label_even = Text("Even (y=x²)", color=COLOR_EVEN, font_size=20)
        self.place_at_grid(label_even, "A2")
        
        parabola_right = axes_even.plot(lambda x: x**2, x_range=[0, 2], color=COLOR_EVEN)
        parabola_left = axes_even.plot(lambda x: x**2, x_range=[-2, 0], color=COLOR_EVEN)
        
        self.play(Create(axes_even), Write(label_even))
        self.play(Create(parabola_right))
        
        # Mirroring animation: transform a copy across the y-axis
        parabola_mirror = parabola_right.copy()
        # Scale by -1 on x-axis relative to axis center
        mirror_matrix = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.play(
            parabola_mirror.animate.apply_matrix(mirror_matrix, about_point=axes_even.c2p(0, 0, 0)),
            run_time=1.5
        )
        self.add(parabola_left)
        self.remove(parabola_mirror)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Odd functions exhibit rotational symmetry through the origin.
        self.play(self.lecture[2].animate.set_color(COLOR_ODD))
        
        axes_odd = Axes(
            x_range=[-2, 2, 1],
            y_range=[-4, 4, 2],
            x_length=2.0,
            y_length=1.5,
            axis_config={"include_tip": False, "color": GREY, "stroke_width": 2}
        )
        self.place_in_area(axes_odd, "B4", "C6")
        
        label_odd = Text("Odd (y=x³)", color=COLOR_ODD, font_size=20)
        self.place_at_grid(label_odd, "A5")
        
        cubic_right = axes_odd.plot(lambda x: x**3, x_range=[0, 1.5], color=COLOR_ODD)
        cubic_left = axes_odd.plot(lambda x: x**3, x_range=[-1.5, 0], color=COLOR_ODD)
        
        self.play(Create(axes_odd), Write(label_odd))
        self.play(Create(cubic_right))
        
        # Rotational symmetry animation: rotate 180 degrees through origin
        cubic_rotation = cubic_right.copy()
        self.play(
            Rotate(cubic_rotation, angle=PI, about_point=axes_odd.c2p(0, 0, 0)),
            run_time=1.5
        )
        self.add(cubic_left)
        self.remove(cubic_rotation)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Cosines are the "Even Kings" of the wave world.
        self.play(self.lecture[3].animate.set_color(COLOR_EVEN))
        
        axes_cos = Axes(
            x_range=[-PI, PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=2.0,
            y_length=1.5,
            axis_config={"include_tip": False, "color": GREY, "stroke_width": 2}
        )
        self.place_in_area(axes_cos, "E1", "F3")
        
        label_cos = Text("Even King: Cosine", color=COLOR_EVEN, font_size=18)
        # Fix for issue 48: scale factor 0.7 for long label in row D
        self.place_at_grid(label_cos, "D2", scale_factor=0.7)
        
        cosine_graph = axes_cos.plot(lambda x: np.cos(x), color=COLOR_EVEN)
        
        self.play(Create(axes_cos), Write(label_cos))
        self.play(Create(cosine_graph))
        self.play(Flash(cosine_graph, color=COLOR_EVEN, flash_radius=1.2, num_lines=12))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Sines serve as the "Odd Queens" in our series.
        self.play(self.lecture[4].animate.set_color(COLOR_ODD))
        
        axes_sin = Axes(
            x_range=[-PI, PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=2.0,
            y_length=1.5,
            axis_config={"include_tip": False, "color": GREY, "stroke_width": 2}
        )
        self.place_in_area(axes_sin, "E4", "F6")
        
        label_sin = Text("Odd Queen: Sine", color=COLOR_ODD, font_size=18)
        # Fix for issue 48: scale factor 0.7 for long label in row D
        self.place_at_grid(label_sin, "D5", scale_factor=0.7)
        
        sine_graph = axes_sin.plot(lambda x: np.sin(x), color=COLOR_ODD)
        
        self.play(Create(axes_sin), Write(label_sin))
        self.play(Create(sine_graph))
        self.play(Flash(sine_graph, color=COLOR_ODD, flash_radius=1.2, num_lines=12))
        self.wait(2)
