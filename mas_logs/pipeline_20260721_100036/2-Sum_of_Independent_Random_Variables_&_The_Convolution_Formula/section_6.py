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

class Section6Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "Summary and Real-World Check"
        lecture_lines = [
            "Summing independent variables requires calculating a convolution.",
            "Remember the process: identify, flip-and-slide, and integrate.",
            "This tool helps engineers analyze noise and project durations."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Summing independent variables requires calculating a convolution.
        # Icons for 'Turtle' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/turtle.svg] (#90EE90) and 'Hare' (#FFD700) tasks appear.
        
        turtle_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turtle.svg").set_color("#90EE90").set_height(0.8)
        turtle_label = Text("Turtle Task", font_size=18, color="#90EE90")
        turtle_icon = VGroup(turtle_svg, turtle_label.next_to(turtle_svg, DOWN, buff=0.1))
        
        hare_box = RoundedRectangle(corner_radius=0.1, height=0.8, width=0.8, color="#FFD700", fill_opacity=0.3)
        hare_label = Text("Hare Task", font_size=18, color="#FFD700")
        hare_icon = VGroup(hare_box, hare_label.next_to(hare_box, DOWN, buff=0.1))
        
        # Issue 51: Move turtle_icon to B3
        self.place_at_grid(turtle_icon, "B3", scale_factor=0.7)
        self.place_at_grid(hare_icon, "B5", scale_factor=0.7)
        
        self.play(self.lecture[0].animate.set_color("#90EE90"))
        self.play(FadeIn(turtle_icon), FadeIn(hare_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Remember the process: identify, flip-and-slide, and integrate.
        # Combine their PDF curves using the 'Flip and Slide' motion.
        
        pdf_turtle = Rectangle(height=0.8, width=1.2, color="#90EE90", fill_opacity=0.4)
        pdf_hare = Rectangle(height=0.8, width=1.2, color="#FFD700", fill_opacity=0.4)
        
        pdf_t_label = MathTex("f_X(x)", font_size=24, color="#90EE90")
        pdf_h_label = MathTex("f_Y(y)", font_size=24, color="#FFD700")
        
        pdf_t_group = VGroup(pdf_turtle, pdf_t_label.next_to(pdf_turtle, UP, buff=0.1))
        pdf_h_group = VGroup(pdf_hare, pdf_h_label.next_to(pdf_hare, UP, buff=0.1))
        
        # Issue 51: Move pdf_t_group to D3
        self.place_at_grid(pdf_t_group, "D3", scale_factor=0.9)
        self.place_at_grid(pdf_h_group, "D5", scale_factor=0.9)
        
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(Create(pdf_t_group), Create(pdf_h_group))
        self.wait(0.5)
        
        # Step: Flip
        flip_label = Text("FLIP", font_size=24, color=WHITE)
        # Use midpoint between D4 and D5 (since targets are D3 and D5)
        flip_pos = (self.grid["D4"] + self.grid["D5"]) / 2
        flip_label.move_to(flip_pos + UP*0.5)
        
        self.play(Write(flip_label))
        self.play(pdf_h_group.animate.rotate(PI, axis=UP))
        self.play(FadeOut(flip_label))
        
        # Step: Slide
        slide_label = Text("SLIDE", font_size=24, color=WHITE)
        slide_label.move_to(flip_pos + UP*0.5)
        
        self.play(Write(slide_label))
        # Slide pdf_h_group to overlap with pdf_t_group at D3
        self.play(
            pdf_h_group.animate.move_to(self.grid["D3"]),
            run_time=2,
            rate_func=linear
        )
        self.play(FadeOut(slide_label))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # This tool helps engineers analyze noise and project durations.
        # Final resulting PDF for total time flashes in #FFFFFF.
        
        res_pdf = Polygon(
            [-1.2, -0.6, 0], [0, 0.6, 0], [1.2, -0.6, 0],
            color=WHITE, fill_opacity=0.6, stroke_width=4
        )
        res_label = MathTex("f_{X+Y}(z)", font_size=28, color=WHITE)
        res_group = VGroup(res_pdf, res_label.next_to(res_pdf, UP, buff=0.1))
        
        # Issue 51: Place res_group in area E3 to F5, scale 0.8
        self.place_in_area(res_group, "E3", "F5", scale_factor=0.8)
        
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeIn(res_group))
        self.play(Flash(res_pdf, color=WHITE, flash_radius=1.2))
        self.wait(3)
