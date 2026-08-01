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
        self.setup_layout("Application: The Antiderivative Shortcut", [
            "To find area, use the function's antiderivative.",
            'Let capital F be the antiderivative of f.',
            'Area is F at b minus F at a.',
            'For Zippy, speed 2t becomes distance t squared.',
            'Plug in the endpoints to find distance covered.'
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Evaluation Theorem: Replacing MathTex with Text to avoid LaTeX dependency error
        ftc_formula = Text("∫ f(x) dx = F(b) - F(a)", color="#FFFFFF", font_size=32)
        self.place_in_area(ftc_formula, "A2", "A5", scale_factor=0.9)
        self.play(Write(ftc_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Plot velocity v(t) = 2t and shade the region from t=0 to t=3.
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 8, 2],
            axis_config={"include_tip": True, "font_size": 18, "label_constructor": Text},
            x_axis_config={"numbers_to_include": [1, 2, 3]},
            y_axis_config={"numbers_to_include": [2, 4, 6, 8]}
        )
        graph = axes.plot(lambda t: 2*t, x_range=[0, 3.5], color="#FF4500")
        area_region = axes.get_area(graph, x_range=[0, 3], color="#FF4500", opacity=0.3)
        v_label = Text("v(t) = 2t", color="#FF4500", font_size=24)
        v_label.next_to(axes.c2p(2, 4), UR, buff=0.1)
        
        plot_group = VGroup(axes, graph, area_region, v_label)
        # Adjusted position based on Issue 46/56
        self.place_in_area(plot_group, "C2", "F4", scale_factor=0.7)
        
        self.play(Create(axes), Create(graph))
        self.play(FadeIn(area_region), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Display the antiderivative F(t) = t^2 using Text
        antid_label = Text("F(t) = t²", color="#00BFFF")
        # Adjusted position and scale based on Issue 48/56
        self.place_in_area(antid_label, "B4", "B6", scale_factor=0.9)
        self.play(Write(antid_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Show step-by-step numerical calculation using a VGroup of Text
        calc_tex = VGroup(
            Text("F(3) - F(0)", font_size=28),
            Text("=", font_size=28),
            Text("9 - 0", font_size=28),
            Text("=", font_size=28),
            Text("9", font_size=28)
        ).arrange(RIGHT, buff=0.2).set_color("#FFFFFF")
        
        # Adjusted position based on Issue 47/56
        self.place_in_area(calc_tex, "E5", "F6", scale_factor=0.8)
        self.play(Write(calc_tex))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        # Move the result '9' into the center of the shaded area and flash it
        moving_9 = calc_tex[4].copy()
        target_center = area_region.get_center()
        
        self.play(
            moving_9.animate.move_to(target_center).set_color("#00FF00").scale(1.5)
        )
        self.play(Flash(moving_9, color="#00FF00", flash_radius=0.4))
        self.wait(2)
